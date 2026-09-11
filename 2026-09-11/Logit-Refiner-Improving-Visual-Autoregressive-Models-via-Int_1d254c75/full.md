# Logit Refiner: Improving Visual Autoregressive Models via Intra-Scale Dependency Modeling

Meimingwei Li<sup>⋆,1</sup>, Stefan Andreas Baumann<sup>⋆,1,2</sup>, Felix Krause<sup>1,2</sup>, and Björn Ommer<sup>1,2</sup>

<sup>1</sup> CompVis @ LMU Munich, Germany 2 Munich Center for Machine Learning (MCML)

Abstract. Visual Autoregressive Models (VAR) generate images through next-scale prediction, producing all tokens within each scale in parallel. We show that this parallel decoding constitutes a mean-field-style approximation that discards spatial dependencies among same-scale tokens, causing locally incoherent samples regardless of backbone capacity – a limitation of the decoding rule. Addressing this limitation, we introduce the Logit Refiner, a lightweight autoregressive module that restores intra-scale dependencies by sequentially sampling tokens conditioned on frozen backbone features. Adding only ∼10% parameters and less than 5% of the base model’s training compute, it plugs into any pretrained VAR checkpoint without retraining. Controlled ablations isolate joint intra-scale sampling – rather than additional capacity or training – as the critical ingredient. Across backbones from 310M to 2B parameters on class-conditional ImageNet 256×256, the refiner consistently improves generation quality, enabling a 1.1B-parameter model to surpass one twice its size. The approach further generalizes to text-to-image generation, confirming that the mean-field bottleneck persists across VAR variants and is efectively alleviated by our method.

Project page: https://compvis.github.io/logit-refiner/.

## Introduction

Autoregressive (AR) modeling achieves remarkable success in language [1–3,v 5, 7, 14, 28, 68, 71, 74] by generating tokens sequentially from a learned joint<sup>i</sup> <sub>distribution. Extending this paradigm to images, however, is challenging: images</sub>X lack a canonical ordering [46, 50, 62, 79], and fully sequential pixel generation [46,a 50,62,79] is prohibitively slow, as sequence lengths quickly reach tens of thousands of tokens.

Visual Autoregressive Modeling (VAR) [73] resolves this by leveraging nextscale prediction, generating images in a coarse-to-fine manner, while predicting all tokens within each scale in parallel. This design yields strong [61] generation results, scaling behavior, and substantially improved eficiency compared to token-wise autoregressive models.

(a) Fixing Coherence in Scale-wise AR  
![](images/e137655a229f55722050ef29fd8795d6b4c568b9e3c94b0c75e2094f65bce7a0.jpg)

![](images/191e188796a72724391e7b0cff69c2c16ed2ea8941bb7187ab0d079d6a7577de.jpg)  
Fig. 1: (a) VAR [73] (top) and scaled-up variants like Infinity [24] (bottom) often generate spatially incoherent samples. Our Logit Refiner addresses this problem via a lightweight add-on to existing pretrained models. (b) The refiner consistently improves scaling behavior, shifting the VAR scaling curve downward, until saturating at the same FID that true unseen samples (validation set) achieve.

Despite accurate per-token predictions, VAR samples often exhibit local spatial incoherence: neighboring patches display mismatched textures, structural discontinuities, or implausible combinations – even when each individual prediction is plausible (Fig. 5). This raises a fundamental question:

## If the per-token marginals are correct, why are the joint samples incoherent?

We identify the root cause as an implicit mean-field-style assumption in VAR’s parallel within-scale decoding, which factorizes the conditional joint into independent per-token distributions, discarding spatial dependencies among tokens at the same scale. This approximation produces token combinations that are individually likely, yet jointly inconsistent. A minimal checkerboard example (Fig. 2) makes this concrete: correct per-pixel probabilities still yield invalid global patterns under independent sampling.

Crucially, this limitation lies in the decoding rule – even predicting the correct pointwise conditional distributions can not yield valid samples in practice, since tokens are decoded independently. The missing component in scale-wise autoregressive image generation is therefore joint within-scale sampling that accounts for the intra-scale token dependencies. We introduce the Logit Refiner, a lightweight add-on autoregressive module that restores intra-scale dependencies while leaving the VAR backbone untouched. It can directly be applied on top of an already pretrained VAR model, without any need for adaptation of the pretrained weights to obtain the quality benefits. Conditioned on the backbone’s alreadycomputed hidden states, the refiner samples tokens sequentially within each scale, requiring only a small causal model to capture the residual dependencies that independent decoding ignores. The module adds only ∼10% parameters, trains in hours (<5% extra training compute) with the backbone frozen, and plugs into any pretrained VAR model with modest inference overhead.

Across backbone sizes from 310M to 2B parameters on class-conditional ImageNet, the Logit Refiner consistently improves generative performance by a wide margin, enabling VAR-d24 + Refiner (1.1B parameters) to surpass the twice as large VAR-d30 (2B). Controlled ablations confirm that these gains stem from dependency modeling rather than additional capacity or training: an architecturematched refiner with bidirectional attention and independent sampling fails to match the autoregressive variant, isolating joint intra-scale sampling as the critical ingredient. The approach also generalizes to text-to-image generation, where the refiner yields consistent improvements on a 2B-parameter model.

Our work makes the following main contributions: starting by tracing the spatial incoherence observed in VAR samples to a specific cause – the mean-fieldstyle approximation inherent in parallel within-scale decoding, which generates tokens independently regardless of the backbone’s capacity – we show that autoregressive within-scale sampling is the minimal correction needed to remove this approximation error, reframing the problem from model capacity to the decoding rule. Then, we introduce a lightweight autoregressive refiner that implements this correction as a plug-in module over frozen backbone features, consistently improving generation quality across model scales from 310M to 2B parameters.

## 2 Related Work

Autoregressive Image Generation. Autoregressive models are a dominant paradigm for image generation, powering many frontier foundation models [21,22, 48,49,82]. Early approaches modeled images as pixel-level sequences [46,47,62,72], whereas modern methods [10, 32, 55, 65, 73, 84, 85] operate on discrete tokens from learned tokenizers [18, 56, 75, 86] and explore alternatives to the standard “sweep” ordering [18, 83, 87], the grouping of multiple tokens [8, 58, 73, 80], or shared backbones with large language models [11, 66, 67, 70, 78, 81]. Our work is orthogonal: rather than changing the token order or the backbone, we correct the independence assumption within parallel decoding groups.

Scale-wise Autoregressive Image Generation. VAR [73] introduced nextscale prediction, generating tokens at progressively finer resolutions while sampling all tokens within each scale in parallel. The paradigm has since been extended in many directions: Infinity and Switti scale it to text-to-image generation [24, 41, 76]; M-VAR [59] and MVAR [88] introduce more eficient backbones, and HMAR [31] combines next-scale prediction with masked autoregressive modeling [8]; FVAR [38], HART [69], and FlowAR [57] alter the prediction target or token representation; and a growing line of work reduces inference cost through token pruning [12, 23, 40], frequency- or entropy-guided skipping [9, 34, 39, 89], and speculative decoding [13]. Beyond class-conditional generation, the paradigm has been applied to multimodal modeling [92, 93], image editing [15, 45], restoration [54], super-resolution [53], segmentation [91], and video generation [29, 41]. Across these extensions, tokens within each scale remain at least partially decoded independently, suggesting that the mean-field-style approximation is a structural property of the scale-wise autoregressive paradigm rather than a task-specific limitation of VAR.

Refinement and Post-hoc Correction. Refining initial predictions appears in many forms, from draft-and-revise generation that iteratively improves masked tokens [33, 90] to image generators with explicit refinement stages that re-predict a full generated image [52, 77]. Other methods [cf. 4] propose adversarially optimizing guidance injection into the decoding rule to improve sample quality. Unlike these, we target a specific shortcoming of VAR-style models [73] – the mean-field-style approximation – and correct it with a lightweight add-on inside the generation loop, avoiding repeated full-model passes over the entire image.

## 3 Mean-Field-style Approximation in Scale Autoregression

Visual Autoregressive Modeling (VAR) [73] generates images through nextscale prediction over a hierarchy of discrete token maps. An image is encoded into K scales $\mathbf { r } _ { 1 : K } = \left( \mathbf { r } _ { 1 } , \ldots , \mathbf { r } _ { K } \right)$ , where each $\mathbf { r } _ { k } \in [ V ] ^ { L _ { k } }$ is a sequence of $L _ { k } = h _ { k } w _ { k }$ tokens at spatial resolution $h _ { k } \times w _ { k }$ . VAR models the joint distribution autoregressively across scales:

$$
p ( \mathbf { r } _ { 1 : K } ) = \prod _ { k = 1 } ^ { K } p _ { \theta } ( \mathbf { r } _ { k } \mid \mathbf { r } _ { < k } ) .\tag{1}
$$

At each scale, a transformer backbone $f _ { \theta }$ produces hidden states $\mathbf { h } ^ { ( k ) } = f _ { \theta } ( \mathbf { r } _ { < k } )$ for all token positions in parallel that parametrize per-token categorical distributions. Tokens within each scale are then sampled independently:

$$
p _ { \theta } ( \mathbf { r } _ { k } \mid \mathbf { r } _ { < k } ) \approx \prod _ { i = 1 } ^ { L _ { k } } p _ { \theta } \left( r _ { i } ^ { ( k ) } \mid \mathbf { h } ^ { ( k ) } \right) ,\tag{2}
$$

which constitutes a fully factorized (naive) mean-field-style approximation that ignores spatial dependencies across tokens in the same scale, akin to those typically used in variational inference in physics [30]. Natural images, however, exhibit strong spatial dependencies, implying that the true conditional joint $p _ { \theta } ( \mathbf { r } _ { k } \mid \mathbf { r } _ { < k } )$ might not be modeled faithfully. This can lead to structurally incoherent samples that persist even at the largest model scales (Fig. 5).

Toy Example (Fig. 2). For a simple dataset containing only valid $2 \times 2$ checkerboards (two valid samples), a next-scale model predicts correct per-token marginals, yet independent sampling produces $2 ^ { 4 } = 1 6$ joint outcomes, most of which are invalid. This illustrates the central failure of Eq. (2): correct marginals do not imply correct joint samples, even when conditioning on previous scales.

![](images/eb23bca0bbec59ac7bb9911c6be9f570761e5c03edd675330285a28db0479a35.jpg)  
Fig. 2: Toy Example. (a) Consider a dataset of $2 \times 2$ checkerboards. (b) A VAR-style model can learn correct per-token marginals (gray at $1 ^ { 2 }$ , 50/50 at $2 ^ { 2 } )$ , yet independent sampling yields many invalid joint samples. (c) Our Logit Refiner models remaining dependencies autoregressively, restoring the joint and generating only valid samples.

This limitation arises from the decoding rule, motivating the restoration of the intra-scale joint distribution.

## 4 Logit Refiner

We introduce the Logit Refiner, a lightweight autoregressive module that restores intra-scale dependencies while leaving the pretrained VAR backbone unchanged. The backbone $f _ { \theta }$ continues to compute per-token hidden states in parallel, whereas a separate refiner model $q _ { \phi }$ models the joint distribution of tokens within each scale conditioned on these frozen features. Since the refiner operates purely at the decoding stage and is not constrained by VAR’s mean-field-style factorization, it can implement flexible joint sampling with minimal additional parameters, computation, and training cost.

## 4.1 Restoring Joint Within-Scale Sampling

Since the backbone’s bidirectional intra-scale attention already enables joint reasoning about all token positions per scale, the independence limitation resides in the sampling rule, not the learned representations (Sec. 3). We therefore replace the mean-field-style decoder in Eq. (2) with an autoregressive factorization over tokens within each scale:

$$
q _ { \phi } ( \mathbf { r } _ { k } \mid \mathbf { r } _ { < k } ) = \prod _ { i = 1 } ^ { L _ { k } } q _ { \phi } \left( r _ { i } ^ { ( k ) } \mid r _ { < i } ^ { ( k ) } , \mathbf { h } _ { \leq i } ^ { ( k ) } , \mathbf { r } _ { < k } \right) ,\tag{3}
$$

where $q _ { \phi }$ is a lightweight refiner model. Each token now conditions on both the frozen backbone features ${ \bf h } _ { \leq i } ^ { ( k ) }$ and the previously sampled tokens $r _ { < i } ^ { ( k ) }$ , restoring the intra-scale joint that the parallel mean-field-style decoder discards. This formulation strictly generalizes the original decoder – setting the refiner in Eq. (3) to ignore the autoregressive context directly recovers Eq. (2) – making it a minimal correction that removes the conditional independence assumption without modifying the backbone.

![](images/331e278c749c640edcf71f6aae2b0daf2dcc79164d16d078740924303280eac5.jpg)  
Fig. 3: Logit Refiner Overview. (a) The VAR backbone processes all previous scales and produces hidden states for the current scale in a single parallel forward pass, finally sampling from pointwise posteriors in parallel. As sampling is done independently within each scale, this can lead to mismatched tokens, afecting generation quality. (b) Our logit refiner takes these hidden states and samples tokens autoregressively within the scale, conditioning each prediction on previously sampled tokens. The refiner is a lightweight causal transformer, incurring only a small overhead during generation, while significantly improving sample quality.

Combining Eq. (3) with the scale-wise factorization Eq. (1) yields the full model:

$$
p ( \mathbf { r } _ { 1 : K } ) \approx \prod _ { k = 1 } ^ { K } q _ { \phi } ( \mathbf { r } _ { k } \mid \mathbf { r } _ { < k } ) ,\tag{4}
$$

which preserves VAR’s eficient across-scale generation while restoring withinscale dependencies. Unlike a fully autoregressive model that runs the entire backbone per token, only the lightweight refiner $q _ { \phi }$ operates sequentially on a token level – the expensive backbone computation remains fully parallel. Because the refiner operates on the backbone’s features rather than building context from scratch, it only needs to model residual dependencies, explaining why an extremely small model sufices (Sec. 4.2).

## 4.2 Architecture

The refiner is designed as a strict add-on: it never re-encodes the image and only consumes i) the already-computed backbone hidden states $\mathbf { h } ^ { ( k ) }$ for the current scale, and ii) the previously generated tokens within the current chunk $r _ { < i } ^ { ( k ) }$ . This separation ensures that the expensive backbone forward pass remains fully parallel; only the lightweight refiner runs sequentially within each scale. We describe each component below and illustrate the architecture in Fig. 3.

Inputs. For each position i in scale k, the refiner constructs an input vector from two streams of information: the frozen backbone hidden state $\mathbf { \bar { h } } _ { i } ^ { ( k ) } \in \mathbb { R } ^ { w }$ , and an autoregressive context embedding $\mathbf { c } _ { i } ^ { ( k ) } \in \mathbb { R } ^ { w }$ derived from the previously sampled token:

$$
\begin{array} { r } { \mathbf { c } _ { i } ^ { ( k ) } = \left\{ \begin{array} { l l } { \mathbf { z } _ { \mathrm { s o s } } , } & { i = 1 , } \\ { \mathrm { e m b } _ { \phi } ( r _ { i - 1 } ^ { ( k ) } ) , } & { i > 1 , } \end{array} \right. \quad \mathbf { z } _ { i } ^ { ( k ) } = \mathbf { W } _ { \mathrm { p r o j } } [ \mathbf { h } _ { i } ^ { ( k ) } \parallel \mathbf { c } _ { i } ^ { ( k ) } ] , } \end{array}\tag{5}
$$

with learned $\mathbf { z } _ { \mathrm { s o s } }$ , token embedding em $\mathsf { \Omega } ) _ { \phi } : [ V ] \to \mathbb { R } ^ { w }$ , and input projection $\mathbf { W } _ { \mathrm { p r o j } }$ , and $[ \cdot \parallel \cdot ]$ denoting concatenation.

Refiner Blocks. A small stack of $d _ { r }$ transformer blocks processes the fused representations $\mathbf { z } _ { 1 : L _ { k } } ^ { ( k ) }$ with a causal mask, producing refined hidden states

$$
\tilde { \mathbf { h } } _ { i } ^ { ( k ) } = \mathrm { T r a n s f o r m e r B l o c k s } _ { \phi } \big ( \mathbf { z } _ { 1 : i } ^ { ( k ) } \big ) .\tag{6}
$$

Each block follows standard transformer design, matching the backbone’s block architecture. Crucially, only $d _ { r } \ll d$ blocks are needed $( \mathrm { e . g . } , d _ { r } = 2 )$ – far fewer than the backbone depth $( d \in [ 1 6 , 3 0 ] )$ , since the refiner already receives a rich, spatially-contextualized hidden state ${ \bf h } _ { i } ^ { ( k ) }$ from the backbone with bidirectional attention. The refiner only needs to model the residual dependencies not captured by the backbone – a much easier task than building spatial context from scratch.

The autoregressive factorization in Eq. (3) requires choosing a token ordering within each scale; we use standard raster-scan order (left-to-right, top-to-bottom), following conventions in patch-level autoregressive models [cf. 46, 47].

Output. An output head predicts logits from the refined hidden state, from which a token is sampled:

$$
\tilde { \ell } _ { i } ^ { ( k ) } = \mathrm { h e a d } _ { \phi } \big ( \tilde { \mathbf { h } } _ { i } ^ { ( k ) } \big ) , \qquad r _ { i } ^ { ( k ) } \sim \mathrm { C a t } \big ( \mathrm { s o f t m a x } \big ( \tilde { \ell } _ { i } ^ { ( k ) } \big ) \big ) .\tag{7}
$$

The sampled token $r _ { i } ^ { ( k ) }$ is then fed back as the autoregressive context for the next position via emb<sub>ϕ</sub> in Eq. (5), and this process repeats sequentially for all $L _ { k }$ positions in the scale.

## 4.3 Training and Inference

Starting from a conventionally pretrained VAR model, we train the refiner with teacher forcing: during training, the autoregressive context $r _ { < i } ^ { ( k ) }$ in Eq. (5) is replaced with ground-truth tokens. Combined with the causal attention mask, this allows training to be fully parallelized across all positions and scales simultaneously. We minimize cross-entropy over all tokens:

$$
\mathcal { L } ( \phi ) = - \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { L _ { k } } \log q _ { \phi } \left( r _ { i } ^ { ( k ) } \mid r _ { < i } ^ { ( k ) } , \mathbf { h } _ { \leq i } ^ { ( k ) } , \mathbf { r } _ { < k } \right) .\tag{8}
$$

During refiner training, the VAR backbone $f _ { \theta }$ remains frozen. We only optimize the refiner parameters $\phi ,$ , finding that this sufices for achieving significant performance gains while keeping training cost minimal.

Identity Initialization. We design an initialization scheme that makes the refiner reproduce the base model’s predictions at the start of training, so that optimization focuses entirely on learning the residual corrections needed for joint sampling. Specifically, we copy the pretrained output head and token embedding weights from the base model, providing the refiner with an already well-structured output space. The autoregressive context integration via $\mathbf { W } _ { \mathrm { p r o j } }$ (in Eq. (5)) is initially disabled by setting $\mathbf { W } _ { \mathrm { p r o j } }  [ \mathbf { I } \parallel \mathbf { 0 } ]$ ; similarly, the output projections of each transformer block’s self-attention and feedforward networks are zeroinitialized to leave the initial hidden states unchanged. Under this scheme, the model at initialization is functionally equivalent to the original VAR, and the training signal drives the refiner to learn only the diference between independent and joint within-scale distributions. As shown in Fig. 4, identity initialization retains the base model’s generation quality from the first iteration and converges significantly faster than standard random initialization.

![](images/a23141306c3e9592035a0ef5d58f5a3c6344241c300e2502cb375cef45a35082.jpg)

Inference. At inference, VAR generates tokens scale-wise in a coarse-to-fine manner. For each scale, we compute $\mathbf { h } ^ { ( k ) }$ once in parallel using the expensive base model $f _ { \theta }$ . Then, we sample tokens $r _ { 1 } ^ { ( k ) } , \ldots , r _ { L _ { k } } ^ { ( k ) }$ sequentially using KV caching in the refiner. Since the refiner is much smaller than the base model $( d _ { r } \ll d )$ , the additional cost is modest and significantly lower than traditional full-model tokenwise autoregressive sampling.

(a) Random Init.

(b) Identity Init.

Fig. 4: Efect of Initialization Strategy.

## 5 Experiments

Experiment Settings. We conduct our experiments on class-conditional ImageNet [16] at a resolution of $2 5 6 ^ { 2 }$ unless noted otherwise, and primarily evaluate the Fréchet Inception Distance (FID) [25] on 50k generated samples. Our base model implementation and training setting directly follow VAR [73], except for a significantly reduced training duration of 30 epochs compared to multiple hundred for the base models, and a reduced base learning rate $( 1 \mathrm { e } { - } 4 \to 2 . 5 \mathrm { e } { - } 5 )$ . Unless noted otherwise, we use $d _ { r } = 2$ refiner layers whose width is matched to the base model’s parameters. For sampling, we use classifier-free guidance (CFG) [27] and top-k sampling following the original VAR settings, with individually swept parameters. Full hyperparameters and details are provided in Supp. Sec. A.

## 5.1 Ablation Studies

We first validate the key design decisions of the Logit Refiner through controlled ablations, establishing that the gains originate from dependency modeling, before presenting final results.

Dependency Modeling, not Capacity, Drives Gains. Table 1 disentangles the contribution of joint intra-scale modeling from additional capacity and training on the same pretrained VAR-d16 backbone, exploring the following variations:

Table 1: What drives the gains? Neither additional training nor additional parallel layers match the improvement from autoregressive (joint) intra-scale refinement. All variants use VAR-d16 as backbone, the “parallel” and ${ } ^ { 6 6 } \mathrm { A R } ^ { 9 }$ refiner use the same architecture $( d _ { r } = 2 )$ , with only a diferent attention mask and sampling. Only causal attention with autoregressive sampling – i.e., joint intra-scale modeling – yields substantial gains.
<table><tr><td>Model</td><td>Joint Modeling Params FID↓</td></tr><tr><td>VAR-d16 (baseline) [73]</td><td>x</td></tr><tr><td>+ Additional Training (30ep)</td><td>310M</td></tr><tr><td>+ Parallel Refiner (bidirectional attention)</td><td>356M</td></tr><tr><td>+ AR Refiner (causal attention, ours)</td><td>3.15 356M 2.81</td></tr></table>

Table 2: Refiner Design Ablation. We start from VAR-d16 [73]. (a) Refiner depth: $d _ { r } = 0$ (no transformer blocks) already improves FID significantly; $d _ { r } = 2$ saturates quality. (b) Trainable components: the refiner alone captures most gains; jointly training the backbone adds little at much higher cost. (c) Per-scale importance: removing the refiner from any single scale degrades quality, most strongly at early scales; degradations diminish at high resolutions.  
(c) Scales with Refiner

(a) Refiner Depth  
(b) Trainable Components
<table><tr><td colspan="2">Refiner Depth (dr) Params FID↓</td><td></td></tr><tr><td></td><td>310M 3.30</td><td></td></tr><tr><td>0</td><td>+8M 3.02</td><td></td></tr><tr><td>1</td><td>+27M 2.85</td><td></td></tr><tr><td>2</td><td>+46M 2.81</td><td></td></tr><tr><td>4</td><td>+84M 2.82</td><td></td></tr><tr><td>8</td><td>+160M 2.81</td><td></td></tr></table>

<table><tr><td></td><td>Trained?</td><td></td><td rowspan="2">Trainable Params</td><td rowspan="2">FID↓</td></tr><tr><td></td><td>Output Head Embedding Base Backbone</td><td></td></tr><tr><td>Baseline</td><td></td><td></td><td>310M</td><td>3.30</td></tr><tr><td>√</td><td>x</td><td>x</td><td>46M</td><td>2.86</td></tr><tr><td>x</td><td>√</td><td>x</td><td>40M</td><td>2.85</td></tr><tr><td>√</td><td>√</td><td>x</td><td>46M</td><td>2.81</td></tr><tr><td>√</td><td>√</td><td>√</td><td>356M</td><td>2.72</td></tr></table>

<table><tr><td>Scales</td><td>FID↓</td><td>Scales</td><td>FID↓</td></tr><tr><td>all</td><td>2.81</td><td>all \ {62}</td><td>2.85</td></tr><tr><td>all  $\scriptstyle { \mathrm { ~ \left\{ ~ 2 ^ { 2 } \right\} ~ } }$ </td><td>2.98</td><td>all\{82}</td><td>2.81</td></tr><tr><td>all  $\dot { | } \dot { \{ \beta ^ { 2 } \} }$ </td><td>2.90</td><td>all  $\dot { { \{ \{ 1 0 ^ { 2 } \} } } $ </td><td>2.85</td></tr><tr><td>all\ {42} 2.86</td><td></td><td>all  $\backslash \{ 1 3 ^ { 2 } \}$ </td><td>2.86</td></tr><tr><td>all \ {52} 2.86</td><td></td><td> $\mathrm { a l l } \setminus \{ 1 6 ^ { 2 } \}$ </td><td>2.83</td></tr><tr><td>[ctd. →]</td><td></td><td></td><td>3.30</td></tr></table>

1. Additional Training: continue training the base model for 30 more epochs without any architectural changes.

2. Parallel Refiner: add refiner-sized transformer blocks with bidirectional attention to the frozen backbone, matching our method’s architecture and parameter count but sampling all tokens independently per scale, isolating capacity from joint modeling.

3. AR Refiner (Ours): identical architecture with causal attention and autoregressive sampling, restoring intra-scale dependencies.

All three variants are trained for the same number of epochs. Only the full AR refiner yields substantial gains, confirming that dependency modeling – not capacity or extra training – is the critical ingredient.

Small Refiners Sufice. Table 2a varies the number of refiner transformer blocks $\left( d _ { r } \right)$ . Even a depth-0 refiner (no additional transformer blocks, just the autoregressive input projection and finetuned head with AR sampling) provides a meaningful improvement. This demonstrates that the autoregressive factorization itself is valuable, even when combined with just a causal context of one token and a single additional linear layer to incorporate extra information. Performance saturates quickly, with $d _ { r } = 2$ providing a favorable tradeof.

![](images/e885b8568a877be5fdd8633d3f204049b8066736a63a036c023304de7ed105a3.jpg)  
Fig. 5: VAR Failure Cases vs. Refiner. Our Logit Refiner can address a range of typical failures of VAR. Each column shows a paired sample (same class, same seed). Top: samples covering various failure modes from VAR-d30 [73]. Bottom: with refiner.

Trainable Components. Table 2b ablates which components need to be trained. Reusing the frozen input embedding or output head from the base VAR model provides a small performance regression compared to training the whole refiner jointly, indicating that the backbone’s learned representations already carry the relevant information – the refiner merely needs to model the residual dependencies that independent sampling discards. Jointly training the VAR backbone yields only minor further gains (FID 2.72) at greatly increased training cost, so we keep the backbone frozen throughout.

Which Scales Benefit Most? Table 2c measures each scale’s contribution by applying the refiner at all but one scale during sampling. The FID degradation relative to full-refiner sampling quantifies how much joint modeling at that scale matters for generation quality. The largest drops occur at the earliest scales. This is notable, as the refiner’s computational cost is also lowest at these scales, suggesting that selectively applying the refiner only at early scales could reduce inference overhead with minimal quality loss (Sec. 5.2).

Collectively, these ablations confirm that the Logit Refiner’s gains stem from restoring intra-scale dependencies, not from additional capacity or training, and that the module is robust across architectural choices, with eficient add-on training on a frozen pretrained backbone suficient to capture significant gains.

## 5.2 Main Results

ImageNet Generation. We identify three recurring failure modes of VAR that persist even at multi-billion parameter scales (Fig. 5-top): samples that consist of class-relevant textures with little obvious structure (“texture soup”), images with multiple inconsistent, often merged instances of the target class, and samples with locally inconsistent structure. All three stem from the lack of spatial coordination inherent in independent within-scale sampling (Sec. 3).

Table 3: System-level Comparison on class-conditional ImageNet-256<sup>2</sup> across discrete-token scale-wise autoregressive models. Within each backbone scale, VAR + Refiner achieves the best FID, improving by 0.16 to 0.49 over VAR with only ∼10% additional parameters. See Supp. Sec. B for additional models & evals w/o CFG.
<table><tr><td>Method</td><td>Params</td><td>FID↓</td><td>IS↑</td><td>Prec↑ Rec↑</td></tr><tr><td>MVAR-d16 [88]</td><td>310M</td><td>3.09</td><td>285.5 0.85</td><td>0.51</td></tr><tr><td>M-VAR-d16 [59]</td><td>464M</td><td>3.07</td><td>294.6 0.84</td><td>0.53</td></tr><tr><td>HMAR-d16 [31]</td><td>465M</td><td>3.01</td><td>288.6 0.84</td><td>0.55</td></tr><tr><td>VAR-d16 [73]</td><td>310M</td><td>3.30</td><td>274.4 0.84</td><td>0.51</td></tr><tr><td>+ Refiner (Ours)</td><td>356M</td><td>2.81v0.49</td><td>267.2 0.81</td><td>0.56</td></tr><tr><td>HART-d20 [69]</td><td>649M</td><td>2.39</td><td>316.4</td><td></td></tr><tr><td>MVAR-d20 [88]</td><td>600M</td><td>2.87</td><td>295.3 0.86</td><td>0.52</td></tr><tr><td>M-VAR-d20 [59]</td><td>900M</td><td>2.41</td><td>308.4 0.85</td><td>0.58</td></tr><tr><td>HMAR-d20 [31]</td><td>840M</td><td>2.50</td><td>319.0 0.85</td><td>0.57</td></tr><tr><td>VAR-d20 [73]</td><td>600M</td><td>2.57</td><td>302.6 0.83</td><td>0.56</td></tr><tr><td>+ Refiner (Ours)</td><td>671M</td><td></td><td>2.17v0.40 274.7 0.80</td><td>0.60</td></tr><tr><td colspan="3">ImageNet Validation</td><td></td><td></td></tr></table>

<table><tr><td>Method</td><td>Params</td><td>FID↓</td><td>IS↑ Prec↑ Rec↑</td><td></td></tr><tr><td>HART-d24 [69]</td><td>1.0B</td><td>2.00</td><td>331.5</td><td></td></tr><tr><td>FastVAR-d24 [23]</td><td>1.0B</td><td>2.64</td><td>287.4 0.80</td><td>0.58</td></tr><tr><td>MVAR-d24 [88]</td><td>1.0B</td><td>2.23</td><td>300.1 0.86</td><td>0.52</td></tr><tr><td>M-VAR-d24 [59]</td><td>1.5B</td><td>1.93</td><td>320.7 0.83</td><td>0.59</td></tr><tr><td>HMAR [31]</td><td>1.3B</td><td>2.10</td><td>319.00.83</td><td>0.60</td></tr><tr><td>VAR-d24 [73]</td><td>1.0B</td><td>2.09</td><td>312.90.83</td><td>0.57</td></tr><tr><td>+ Refiner (Ours)</td><td>1.1B</td><td> $1 . 8 3 _ { \overline { { \tau } } 0 . 2 6 }$ </td><td>288.2 0.79</td><td>0.63</td></tr><tr><td>HART-d30 [69]</td><td>2.0B</td><td>1.77</td><td>330.3</td><td></td></tr><tr><td>FastVAR-d30 [23]</td><td>2.0B</td><td>2.30</td><td>288.7 0.81</td><td>0.59</td></tr><tr><td>VAR-CoDe-d30 [13]</td><td>2.3B</td><td>1.94</td><td>296</td><td>0.81 0.60</td></tr><tr><td>HMAR [31]</td><td>2.4B</td><td>1.95</td><td>334.5 0.82</td><td>0.62</td></tr><tr><td>VAR-d30 [73]</td><td>2.0B</td><td>1.92</td><td>323.1 0.82</td><td>0.58</td></tr><tr><td>+ Refiner (Ours)</td><td>2.2B</td><td> $1 . 7 6 \substack { \mathrm { ~ \textmu ~ } _ { \mathrm { ~ 0 ~ . ~ 1 6 ~ } } }$ </td><td>319.4 0.80</td><td>0.62</td></tr><tr><td>M-VAR-d32 [59]</td><td>3.0B</td><td>1.78</td><td>331.2 0.83</td><td>0.61</td></tr></table>

![](images/26ec183f4fe0405a245b39cfe40a607f5daabe6c12e1cba0b8f3d32d1fcab5fa.jpg)

![](images/ccd02d9ed60867f8d655263917a3ffc48fb1d7400ae03ebc38d9eb1dea754e26.jpg)  
IS ( )  
Fig. 6: FID-IS Improvement Tradeof. By varying the classifier-free guidance scale, the logit refiner can achieve improvements in both dimensions.

Adding our refiner enables the model to sample from the joint intra-scale token distribution, directly addressing these failure modes (Fig. 5-bottom). We show additional qualitative samples in Supp. Sec. C.2.

These qualitative gains are reflected in quantitative metrics. Table 3 compares our Logit Refiner applied to VAR across scales and with a broad range of scale-wise autoregressive methods. Reference results for other generative model families are reported in the extended comparison (Supp. Tab. B.4). Across all model scales, the refiner consistently improves FID by a significant margin (0.16 to 0.49), while only adding ∼10% additional parameters. VAR-d24 + Refiner (1.1B params total) even exceeds VAR-d30 (2B params) by a significant margin, obtaining a stronger model at roughly half the size. Beyond FID, the refiner also consistently improves recall by 0.04 to 0.06 across all backbone scales, indicating that restoring intra-scale dependencies recovers sample diversity rather than trading it away. The small accompanying decrease in Inception Score follows from the lower CFG scales that are FID-optimal for the refiner, not from reduced sample quality – by varying the guidance scale, improvements in FID and/or IS over the baseline can be traded of (see Fig. 6). We also compare in a CFG-free setting in Supp. Tab. B.5, where the logit refiner also consistently outperforms the baseline. Compared with other VAR variants that address orthogonal aspects, the basic VAR model with our refiner achieves the best generative performance within each model scale. Our approach does not utilize any of the improvements introduced by these methods, which may lead to further gains in combination. These directions are left for future work.

VAR Scale  
![](images/8a028a06fe92d169544d2f5325858500dde4f0c80c867fd3211e5dde41ac5459.jpg)  
Fig. 7: Qualitative Scaling Behavior. Paired samples (same class, seed) across backbone scales (VAR-{16, 20, 24, 30}). Top rows: vanilla VAR improves visual fidelity and class consistency with scale, but spatial consistency problems persist at every size. Bottom rows: adding our refiner resolves these artifacts across all scales.

Scaling Behavior. Figure 7 investigates how the Logit Refiner interacts with backbone scale. Qualitatively, (Fig. 7), spatial consistency problems persist across all vanilla VAR scales, even as visual fidelity improves with model size. The refiner resolves these artifacts at every scale, confirming that the underlying issue is the mean-field-style sampling rule rather than insuficient model capacity. Quantitatively (Fig. 1b), the refiner shifts the FID scaling curve downward across all backbone sizes without altering the overall scaling trend, reflecting the qualitative scaling findings, and demonstrating that correcting the mean-fieldstyle approximation can be more parameter-eficient than scaling the backbone.

Training and Inference Eficiency. The Logit Refiner corrects a samplingtime approximation, modeling residual dependencies between tokens within each scale. This suggests that a refiner trained on a frozen backbone should already capture most of the achievable gains, since the marginals are correct and only the dependencies are missing. Our results confirm this: adding a refiner to a frozen backbone improves the FID from 3.30 to 2.81 with only 66 H200-h of additional training compute. Training the full model with an integrated refiner from scratch yields a stronger FID of 2.57, but requires 1,845 H200-h – 28× more compute for the remaining third of improvement. Jointly finetuning the backbone occupies a middle ground (FID 2.72, 127 H200-h). The dominant efect is thus the correction of the mean-field-style factorization itself, achievable as a lightweight post-hoc addition to any pretrained VAR model without retraining the backbone.

![](images/43548f07a8cf184324dc7bb98d173d3b34b952690779cff9ecb960ee2bbf869c.jpg)  
(a) VAR-d16 stage tradeof

![](images/a0b993961dd7863f3ecf9c9bd311b056fe5aff785b3d9b06c2460d17f6245fa6.jpg)  
(b) Across backbone scales and batch sizes  
Fig. 8: Quality/Eficiency Tradeof. (a) For VAR-d16, applying the refiner only at the first k scales (Tab. 2c) traverses the quality/eficiency tradeof. (b) The same efect across backbone scales $( \mathrm { d } \{ 1 6 , 2 0 , 2 4 , 3 0 \} )$ ) and batch-size regimes. Left: latency-optimized regime (batch size 1, the worst case for the refiner); Right: typical regime (batch size 16). At batch size 16, the refiner Pareto-dominates the baseline at every scale; at batch size 1, it adds intermediate operating points at low depths and dominates from d24/d30.

During inference, the refiner introduces sequential within-scale sampling, but the cost is modest. The expensive backbone forward pass remains fully parallel: for each scale k, the backbone computes all hidden states $\mathbf { h } ^ { ( k ) }$ in a single pass. Only the lightweight refiner blocks $( d _ { r } = 2 $ blocks vs. $d \in [ 1 6 , 3 0 ]$ backbone blocks) run sequentially, and we employ KV caching to avoid redundant computations across tokens within a scale.

The relative overhead remains modest across backbone sizes and batch sizes (Fig. 8b), since the expensive backbone still runs once per scale in parallel and only the two-layer refiner is sequential – the refiner does not turn VAR into a fully token-wise autoregressive model. At a typical batch size, the refiner improves the quality/eficiency Pareto frontier over vanilla VAR at every backbone scale; in the latency-optimized single-sample regime, it adds finer-grained operating points and dominates the baseline from VAR-d24 onward.

Moreover, the scales ablation (Tab. 2c) shows that the refiner’s quality gains concentrate at early scales, which contain the fewest tokens. For VAR-d16, this enables a practical trade-of (Fig. 8a): applying the refiner only at the first few scales can substantially reduce the sequential sampling cost while retaining most of the quality improvement. Concretely, applying the refiner on scales up to $8 ^ { 2 } / 1 0 ^ { 2 }$ reduces the refiner overhead by 84%/71% while retaining 88%/99% of the full FID improvement, respectively.

Scaling to T2I. To validate that the Logit Refiner generalizes beyond classconditional ImageNet, we apply it to Infinity [24], a scaled VAR variant for text-to-image synthesis. We train the refiner on Infinity’s backbone following a similar setup as for VAR (frozen backbone, $d _ { r } \ = \ 2$ , 100k steps at batch size 768; ∼640 H200-h train time – orders of magnitude less than the base model’s pretraining time) on images and captions from FLUX-6M [19]. Following the findings from the previous paragraph, we apply the refiner selectively to the first several stages (up to resolution $6 ^ { 2 } )$ . Qualitatively (Fig. 9; see also Supp. Sec. C.1 for additional examples), the refiner yields the same types of improvements observed on ImageNet: improved structural coherence and reduced texture inconsistencies. These improvements are also reflected in quantitative evaluations: the refiner improves HPSv3 [44] scores from 9.79 to 9.91 (see Tab. 4), confirming that the benefits of restoring intra-scale dependencies transfer to open-vocabulary text-to-image generation.

![](images/8f9c0ab422901f60fb70ba781e606d8e3df327459db76cf8dccc582f457ee84f.jpg)  
Fig. 9: Qualitative Text-to-Image Results. We show results from Infinity-2B [24] at $1 0 2 4 ^ { 2 }$ resolution without (top) and with our refiner (bottom). As in the ImageNet class-conditional case, our Logit Refiner improves spatial coherence (inconsistencies in original images marked in orange) of the generated images.

Table 4: Quantitative T2I Results on HPSv3 [44]. We evaluate automated preference scores (higher is better) for samples generated with and without our Logit Refiner. It improves spatial consistency in generated images, reflected in better scores.  
![](images/e195f124ddf8228f98dea8edd88fcf61fa7a50c713c1754b8d278f6660f74b77.jpg)

## 6 Conclusion

Scale-wise visual autoregressive generation has a fundamental limitation: parallel within-scale decoding constitutes a mean-field-style approximation that discards spatial dependencies among tokens, causing locally incoherent samples regardless of backbone capacity or training duration. The Logit Refiner addresses this by restoring intra-scale dependencies through a lightweight autoregressive module that operates on parallel backbone features, adding only ∼10% parameters and requiring less than 5% of the base model’s training compute. Across a large range of backbone sizes and both class- and text-conditional generation, the refiner consistently improves generation quality by a wide margin. Controlled ablations isolate joint intra-scale sampling rather than additional capacity or training as the critical ingredient.

Limitations and Future Work. Autoregressive within-scale sampling introduces sequential computation at each scale. While applying the refiner selectively at early scales retains the vast majority of quality gains at a fraction of the cost, the overhead is not fully eliminated. More broadly, our results suggest that when parallel decoding introduces mean-field-style assumptions, lightweight autoregressive correction can restore the discarded dependencies at minimal cost – a potentially general principle that shows promise for applications in parallel generative architectures [e.g. 8, 37]. Combining the refiner with orthogonal VAR improvements [e.g. 59, 69, 88] and exploring non-autoregressive approaches to within-scale sampling are further promising directions.

## Acknowledgments

This project has been supported by the Horizon Europe project ELLIOT (GA No. 101214398), the project “GeniusRobot” (01IS24083) funded by the Federal Ministry of Research, Technology and Space (BMFTR), the BMWE ZIM-project (No. KK5785001LO4) “conIDitional LoRA”, the German Federal Ministry for Economic Afairs and Energy within the project “NXT GEN AI METHODS - Generative Methoden für Perzeption, Prädiktion und Planung”, and the bidt project KLIMA-MEMES. The authors gratefully acknowledge the Gauss Center for Supercomputing for providing compute through the NIC on JUWELS/JUPITER at JSC and the HPC resources supplied by the NHR@FAU Erlangen. We thank Ulrich Prestel, Jack Gallagher, Tommaso Martorella, Ming Gui, Nick Stracke, Kolja Bauer, and Vincent Tao Hu for feedback, proofreading, and helpful discussions, and Owen Vincent for technical support.

## References

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

2. Anil, R., Dai, A.M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al.: Palm 2 technical report. arXiv preprint arXiv:2305.10403 (2023)

3. Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al.: Qwen technical report. arXiv preprint arXiv:2309.16609 (2023)

4. Bi, L., Huang, T., Guo, J., Xu, C.: Adversarial error correction for visual autoregressive generation (2026), https://arxiv.org/abs/2605.24843

5. Bi, X., Chen, D., Chen, G., Chen, S., Dai, D., Deng, C., Ding, H., Dong, K., Du, Q., Fu, Z., et al.: Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954 (2024)

6. Brock, A., Donahue, J., Simonyan, K.: Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096 (2018)

7. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901 (2020)

8. Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W.T.: Maskgit: Masked generative image transformer. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11315–11325 (2022)

9. Chen, J., Lin, R., Zheng, Z., Li, J., Li, M., Luo, G., et al.: Toprovar: Eficient visual autoregressive modeling via tri-dimensional entropy-aware semantic analysis and sparsity optimization. arXiv preprint arXiv:2602.22948 (2026)

10. Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., Sutskever, I.: Generative pretraining from pixels. In: III, H.D., Singh, A. (eds.) Proceedings of the 37th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 119, pp. 1691–1703. PMLR (13–18 Jul 2020)

11. Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C.: Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811 (2025)

12. Chen, Z., Fan, J., Yu, Z., Zhuang, B., Tan, M.: Frequency-aware autoregressive modeling for eficient high-resolution image synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17140–17149 (2025)

13. Chen, Z., Ma, X., Fang, G., Wang, X.: Collaborative decoding makes visual autoregressive modeling eficient. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 23334–23344 (2025)

14. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., et al.: Palm: Scaling language modeling with pathways. Journal of machine learning research 24(240), 1–113 (2023)

15. Dao, Q., He, X., Han, L., Nguyen, N.H., Nobar, A.H., Ahmed, F., Zhang, H., Nguyen, V.A., Metaxas, D.: Discrete noise inversion for next-scale autoregressive text-based image editing. arXiv preprint arXiv:2509.01984 (2025)

16. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition. pp. 248–255 (2009). https://doi.org/10.1109/CVPR.2009. 5206848

17. Dhariwal, P., Nichol, A.: Difusion models beat gans on image synthesis. Advances in neural information processing systems 34, 8780–8794 (2021)

18. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12873–12883 (2021)

19. Fang, R., Yu, A., Duan, C., Huang, L., Bai, S., Cai, Y., Wang, K., Liu, S., Liu, X., Li, H.: Flux-reason-6m & prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680 (2025)

20. Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)

21. Google DeepMind, .: Gemini 2.5 Flash Image (Nano Banana). https://deepmind. google/models/gemini-image/flash/ (2025)

22. Google DeepMind, .: Gemini 3 Pro Image (Nano Banana Pro). https://deepmind. google/models/gemini-image/pro/ (2025)

23. Guo, H., Li, Y., Zhang, T., Wang, J., Dai, T., Xia, S.T., Benini, L.: Fastvar: Linear visual autoregressive modeling via cached token pruning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19011–19021 (2025)

24. Han, J., Liu, J., Jiang, Y., Yan, B., Zhang, Y., Yuan, Z., Peng, B., Liu, X.: Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15733–15744 (2025)

25. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

26. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

27. Ho, J., Salimans, T.: Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598 (2022)

28. Hofmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D.d.L., Hendricks, L.A., Welbl, J., Clark, A., et al.: Training compute-optimal large language models. arXiv preprint arXiv:2203.15556 (2022)

29. Ji, L., Liu, X., Shang, J., Wang, S., Sun, Y., Wu, H., Wang, H.: Videoar: Autoregressive video generation via next-frame & scale prediction. arXiv preprint arXiv:2601.05966 (2026)

30. Jordan, M.I., Ghahramani, Z., Jaakkola, T.S., Saul, L.K.: An introduction to variational methods for graphical models. Machine learning 37(2), 183–233 (1999)

31. Kumbong, H., Liu, X., Lin, T.Y., Liu, M.Y., Liu, X., Liu, Z., Fu, D.Y., Re, C., Romero, D.W.: Hmar: Eficient hierarchical masked auto-regressive image generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2535–2544 (2025)

32. Lee, D., Kim, C., Kim, S., Cho, M., Han, W.S.: Autoregressive image generation using residual quantization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11523–11532 (2022)

33. Lee, D., Kim, C., Kim, S., Cho, M., HAN, W.S.: Draft-and-revise: Efective image generation with contextual rq-transformer. Advances in Neural Information Processing Systems 35, 30127–30138 (2022)

34. Li, J., Ma, Y., Zhang, X., Wei, Q., Liu, S., Zhang, L.: Skipvar: Accelerating visual autoregressive modeling via adaptive frequency-aware skipping. arXiv preprint arXiv:2506.08908 (2025)

35. Li, T., Chang, H., Mishra, S., Zhang, H., Katabi, D., Krishnan, D.: Mage: Masked generative encoder to unify representation learning and image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2142–2152 (2023)

36. Li, T., He, K.: Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720 (2025)

37. Li, T., Tian, Y., Li, H., Deng, M., He, K.: Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems 37, 56424– 56445 (2024)

38. Li, X., Wu, C., Sun, Y., Zhou, J., Qu, D., Qu, Y., Bo, W., Yu, H., Liang, D.: Fvar: Visual autoregressive modeling via next focus prediction. arXiv preprint arXiv:2511.18838 (2025)

39. Li, Y., Wang, H., et al.: Freqexit: Enabling early-exit inference for visual autoregressive models via frequency-aware guidance. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)

40. Li, Z., Wang, N., Bai, T., Mei, C., Wang, P., Qiu, S., Cheng, J.: Sparvar: Exploring sparsity in visual autoregressive modeling for training-free acceleration. arXiv preprint arXiv:2602.04361 (2026)

41. Liu, J., Han, J., Yan, B., Wuhui, Zhu, F., Wang, X., Jiang, Y., PENG, B., Yuan, Z.: Infinitystar: Unified spacetime autoregressive modeling for visual generation. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)

42. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

43. Ma, N., Goldstein, M., Albergo, M.S., Bofi, N.M., Vanden-Eijnden, E., Xie, S.: Sit: Exploring flow and difusion-based generative models with scalable interpolant transformers. In: European Conference on Computer Vision. pp. 23–40. Springer (2024)

44. Ma, Y., Wu, X., Sun, K., Li, H.: Hpsv3: Towards wide-spectrum human preference score. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15086–15095 (2025)

45. Mao, Q., Cai, Q., Li, Y., Pan, Y., Cheng, M., Yao, T., Liu, Q., Mei, T.: Visual autoregressive modeling for instruction-guided image editing. arXiv preprint arXiv:2508.15772 (2025)

46. Van den Oord, A., Kalchbrenner, N., Espeholt, L., Vinyals, O., Graves, A., et al.: Conditional image generation with pixelcnn decoders. Advances in neural information processing systems 29 (2016)

47. Oord, A.v.d., Kalchbrenner, N., Kavukcuoglu, K.: Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759 (2016)

48. OpenAI, .: Gpt image 1. https://developers.openai.com/api/docs/models/gptimage-1 (2025)

49. OpenAI, .: Gpt image 1.5. https://developers.openai.com/api/docs/models/ gpt-image-1.5 (2025)

50. Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, L., Shazeer, N., Ku, A., Tran, D.: Image transformer. In: International conference on machine learning. pp. 4055–4064. PMLR (2018)

51. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)

52. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: SDXL: Improving latent difusion models for high-resolution image synthesis. In: The Twelfth International Conference on Learning Representations (2024)

53. Qu, Y., Yuan, K., Hao, J., Zhao, K., Xie, Q., Sun, M., Zhou, C.: Visual autoregressive modeling for image super-resolution. arXiv preprint arXiv:2501.18993 (2025)

54. Rajagopalan, S., Narayan, K., Patel, V.M.: Restorevar: Visual autoregressive generation for all-in-one image restoration. arXiv preprint arXiv:2505.18047 (2025)

55. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever, I.: Zero-shot text-to-image generation. In: International conference on machine learning. pp. 8821–8831. Pmlr (2021)

56. Razavi, A., Van den Oord, A., Vinyals, O.: Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems 32 (2019)

57. Ren, S., Yu, Q., He, J., Shen, X., Yuille, A., Chen, L.C.: Flowar: Scale-wise autoregressive image generation meets flow matching. arXiv preprint arXiv:2412.15205 (2024)

58. Ren, S., Yu, Q., He, J., Shen, X., Yuille, A., Chen, L.C.: Beyond next-token: Next-x prediction for autoregressive visual generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15781–15791 (2025)

59. Ren, S., Yu, Y., Ruiz, N., Wang, F., Yuille, A., Xie, C.: M-var: Decoupled scalewise autoregressive modeling for high-quality image generation. arXiv preprint arXiv:2411.10433 (2024)

60. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

61. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recognition challenge. International journal of computer vision 115(3), 211–252 (2015)

62. Salimans, T., Karpathy, A., Chen, X., Kingma, D.P.: Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. arXiv preprint arXiv:1701.05517 (2017)

63. Sauer, A., Schwarz, K., Geiger, A.: Stylegan-xl: Scaling stylegan to large diverse datasets. In: ACM SIGGRAPH 2022 conference proceedings. pp. 1–10 (2022)

64. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456 (2020)

65. Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., Yuan, Z.: Autoregressive model beats difusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525 (2024)

66. Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Wang, Y., Rao, Y., Liu, J., Huang, T., Wang, X.: Generative multimodal models are in-context learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14398–14409 (2024)

67. Sun, Q., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, Y., Gao, H., Liu, J., Huang, T., Wang, X.: Emu: Generative pretraining in multimodality. In: The Twelfth International Conference on Learning Representations (2024)

68. Sun, Y., Wang, S., Feng, S., Ding, S., Pang, C., Shang, J., Liu, J., Chen, X., Zhao, Y., Lu, Y., et al.: Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. arXiv preprint arXiv:2107.02137 (2021)

69. Tang, H., Wu, Y., Yang, S., Xie, E., Chen, J., Chen, J., Zhang, Z., Cai, H., Lu, Y., Han, S.: Hart: Eficient visual generation with hybrid autoregressive transformer. arXiv preprint arXiv:2410.10812 (2024)

70. Team, C.: Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818 (2024)

71. Team, G., Anil, R., Borgeaud, S., Alayrac, J.B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A.M., Hauth, A., Millican, K., et al.: Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023)

72. Theis, L., Bethge, M.: Generative image modeling using spatial lstms. Advances in neural information processing systems 28 (2015)

73. Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 84839–84865 (2024)

74. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)

75. Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017)

76. Voronov, A., Kuznedelev, D., Khoroshikh, M., Khrulkov, V., Baranchuk, D.: Switti: Designing scale-wise transformers for text-to-image synthesis. arXiv preprint arXiv:2412.01819 (2024)

77. Wang, J., Zhou, Z., Mummadi, C.K., Dianat, S., Rabbani, M., Rao, R., Qiu, C., Tao, Z.: Visual self-refinement for autoregressive models. arXiv preprint arXiv:2510.00993 (2025)

78. Wang, X., Cui, Y., Wang, J., Zhang, F., Wang, Y., Zhang, X., Luo, Z., Sun, Q., Li, Z., Wang, Y., et al.: Multimodal learning with next-token prediction for large multimodal models. Nature pp. 1–7 (2026)

79. Wang, Y., Yang, S., Zhao, B., Zhang, L., Liu, Q., Zhou, Y., Xie, C.: Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033 (2025)

80. Wang, Y., Ren, S., Lin, Z., Han, Y., Guo, H., Yang, Z., Zou, D., Feng, J., Liu, X.: Parallelized autoregressive visual generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 12955–12965 (2025)

81. Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., et al.: Janus: Decoupling visual encoding for unified multimodal understanding and generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 12966–12977 (2025)

82. xAI, .: Grok imagine. https://docs.x.ai/developers/model-capabilities/ images/generation (2025)

83. Xu, Y., Ju, J., Luan, J., Cui, J.: Direction-aware diagonal autoregressive image generation. arXiv preprint arXiv:2503.11129 (2025)

84. Yu, J., Li, X., Koh, J.Y., Zhang, H., Pang, R., Qin, J., Ku, A., Xu, Y., Baldridge, J., Wu, Y.: Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627 (2021)

85. Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., et al.: Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 2(3), 5 (2022)

86. Yu, L., Lezama, J., Gundavarapu, N.B., Versari, L., Sohn, K., Minnen, D., Cheng, Y., Birodkar, V., Gupta, A., Gu, X., et al.: Language model beats difusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737 (2023)

87. Yu, Q., He, J., Deng, X., Shen, X., Chen, L.C.: Randomized autoregressive visual generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 18431–18441 (2025)

88. Zhang, J., Long, W., Han, M., You, W., Gu, S.: MVAR: Visual autoregressive modeling with scale and spatial markovian conditioning. In: The Fourteenth International Conference on Learning Representations (2026)

89. Zhang, Y., Liu, J., Liu, F., Miao, D., Zhang, Q., Fu, K., Wang, C., Cao, L.: Adaptive visual autoregressive acceleration via dual-linkage entropy analysis. arXiv preprint arXiv:2602.01345 (2026)

90. Zheng, H.K., Li, P.: Lsrs: Latent scale rejection sampling for visual autoregressive modeling. arXiv preprint arXiv:2512.03796 (2025)

91. Zheng, R., Qi, L., Chen, X., Wang, Y., Wang, K., Zhao, H.: Seg-var: Image segmentation with visual autoregressive modeling. arXiv preprint arXiv:2511.12594 (2025)

92. Zhuang, X., Xie, Y., Deng, Y., Liang, L., Ru, J., Yin, Y., Zou, Y.: Vargpt: Unified understanding and generation in a visual autoregressive multimodal large language model. arXiv preprint arXiv:2501.12327 (2025)

93. Zhuang, X., Xie, Y., Deng, Y., Yang, D., Liang, L., Ru, J., Yin, Y., Zou, Y.: Vargptv1. 1: Improve visual autoregressive large unified model via iterative instruction tuning and reinforcement learning. arXiv preprint arXiv:2504.02949 (2025)

## Supplementary Material

## A Implementation Details

Training Hyperparameters. We show relevant hyperparameters for all trained model variations in Supp. Table A.1. Optimizer settings have been directly taken from VAR, with the exception of a reduced learning rate (result of a sweep; we find that the learning rate does not meaningfully influence the FID once converged for our base configuration, with more than double and less than half the learning rate resulting in similar sample metrics) and a linear warmup + cosine decay schedule. For all VAR models, we train for 40 epochs. All variants are converged by that time. For Infinity [24], we train for 200k steps.

Table A.1: Hyperparameters.
<table><tr><td>Variant</td><td>Ablations</td><td>Main Results</td><td>Text-to-Image</td></tr><tr><td>Dataset</td><td>ImageNet-2562</td><td>ImageNet-2562</td><td>FLUX-6M [19]</td></tr><tr><td>Base Model</td><td>VAR-d16 [73]</td><td>VAR-d{16,20,24,30} [73]</td><td>Infinity-2B [24]</td></tr><tr><td>Base Model Depth d</td><td>16</td><td>{16,20,24,30}</td><td>32</td></tr><tr><td>Base Model Width</td><td>1024</td><td>{1024,1280,1536,1920}</td><td>2048</td></tr><tr><td>Refiner Depth d,.</td><td>{0, 1, 2, 4, 8}</td><td>2</td><td>2</td></tr><tr><td>Refiner Width</td><td>1024 (matching base)</td><td>matching base</td><td>2048 (matching base)</td></tr><tr><td>Trainable Parameters</td><td>{8M,27M,46M,84M,160M}</td><td>varying</td><td>193M</td></tr><tr><td>Batch Size</td><td>768</td><td>768 if d &lt; 30, else 1024</td><td>768</td></tr><tr><td>Training Duration</td><td>30 epochs</td><td>30 epochs</td><td>200k steps</td></tr><tr><td>Precision</td><td>fp16 MP</td><td>fp16 MP</td><td>bf16 MP</td></tr><tr><td>Training Hardware</td><td>8 H200</td><td>8-32 H200</td><td>64 H200</td></tr><tr><td>Optimizer</td><td>AdamW [42]</td><td>AdamW [42]</td><td>AdamW [42]</td></tr><tr><td>Base LR (LR = LRbase · BS/256)</td><td>2.5 · 10−5</td><td>2.5 · 10−5</td><td>2.5 · 10−5</td></tr><tr><td>Learning Rate Warmup Learning Rate Schedule</td><td>2% of training from 0.5% of peak</td><td>2% of training from 0.5% of peak</td><td>2% of training from 0.5% of peak</td></tr><tr><td>Betas (β1, β2)</td><td>linear (0.9,0.95)</td><td>linear (0.9, 0.95)</td><td>linear (0.9,0.95)</td></tr><tr><td>Weight Decay</td><td>0.05</td><td></td><td>0.01</td></tr><tr><td></td><td></td><td>0.05; d30: scheduled following VAR [73]</td><td></td></tr></table>

Inference Hyperparameters. We observe that, like with most other image generation models, final performance of VAR + Logit Refiner as measured by FID varies with inference hyperparameters, whose optimal values vary with base model size. We therefore sweep both top-k and CFG [27] scales individually w.r.t. FID. Over CFG, results are generally smooth (i.e., straightforward to sweep, typically close to convex). Supp. Figure A.1 shows the result of our hyperparameter search, where we sweep the CFG scale in increments of 0.1 and k in increments of 100 around the optimum. FID [25] is computed on 50k samples with class-balanced sampling following the baseline VAR [73]. The optimal params we found are listed in Supp. Tab. A.2.

Evaluation Details. For our VAR-d16+2 refiner, we try training with diferent random seeds to estimate confidence intervals for our results. Across four training runs, we obtain the following FIDs: 2.77, 2.80, 2.81, 2.83, resulting in a sample standard deviation of 0.025. This puts our gains compared to all baselines in Tab. 3 to at least 8 standard deviations, indicating that the gains are statistically significant. Across all runs, we do not choose specific checkpoints, but use the same training setup, including seed. For HPSv3, we evaluate the first 50 prompts per subset (600 images total), since sampling all images would be prohibitively expensive.

Table A.2: Inference Hyperparameters. Used for quantitative evaluations. Parameters for Infinity-2B directly mirror those of the base model.
<table><tr><td colspan="5">Base Model VAR-d16 VAR-d20 VAR-d24 VAR-d30 Infinity-2B</td></tr><tr><td>CFG w</td><td>1.8</td><td>1.5</td><td>1.5</td><td>1.9 3.0</td></tr><tr><td>Top-k</td><td>1100</td><td>900</td><td>800 500</td><td>900</td></tr><tr><td>Top-p</td><td></td><td></td><td></td><td>0.97</td></tr><tr><td>T</td><td></td><td></td><td></td><td>0.5</td></tr></table>

![](images/9b56df9c96441129a7e4ece5a7997604e40d755bf61de7b98d45037c98632784.jpg)  
Fig. A.1: Inference Hyperparameter Exploration. The optimal classifier-free guidance scale w and top-k threshold for VAR with our refiner vary across scales.

Within-Scale Token Ordering. We generally use “sweep”/row-major token ordering for the refiner. We explore whether this choice is relevant for final performance by training refiners with five diferent intra-scale orderings in Supp. Tab. A.3. Inference hparams are reused from original “sweep” order; per-order tuning would likely close the small remaining diferences. All tested orderings retain the gain, showing that the improvement is not an artifact of raster scan.

## B Additional Quantitative Evaluation Details

We show an extended version of Tab. 3 in Supp. Tab. B.4. In addition to the major improvement in FID discussed in the main paper, we also observe a minor decrease in IS and precision, and an increase in recall. We attribute the reduction of IS values primarily to VAR with a refiner requiring lower CFG scales w for optimal FID (higher guidance scales are generally associated with higher IS values). Further, we show evaluations without classifier-free guidance in Supp. Tab. B.5: gains with the refiner vs. the baseline persist, and are more pronounced for large models than for small models.

Table A.3: Influence of Intra-Scale Token Ordering. All orderings yield nearidentical gains.
<table><tr><td>Method</td><td>VAR-d16</td><td colspan="2">+ Refiner (Ours)</td></tr><tr><td>Sampling Order</td><td></td><td>parallel sweep (paper) col.-major alternate spiral in spiral out I</td><td>三 口 口</td></tr><tr><td>FID (↓)</td><td>3.30</td><td>2.810.49</td><td>2.83v0.47 2.82v0.48 2.83v0.47 2.86v0.44</td></tr></table>

Table B.4: Extended System-level Comparison on class-conditional ImageNet-256<sup>2</sup> (extending Tab. 3).
<table><tr><td>Method</td><td>Params</td><td>FID↓</td><td>IS↑</td><td>Prec↑ Rec↑</td><td></td></tr><tr><td colspan="6">Scale-wise Autoregression</td></tr><tr><td>MVAR-d16 [88]</td><td>310M</td><td>[73] 3.09</td><td></td><td>285.5 0.85</td><td>0.51</td></tr><tr><td>M-VAR-d16 [59]</td><td>464M</td><td>3.07</td><td>294.6</td><td>0.84</td><td>0.53</td></tr><tr><td>HMAR-d16 [31]</td><td>465M</td><td>3.01</td><td>288.6</td><td>0.84</td><td>0.55</td></tr><tr><td>VAR-d16 [73]</td><td>310M</td><td>3.30</td><td>274.4</td><td>0.84</td><td>0.51</td></tr><tr><td>+ Refiner (Ours)</td><td>356M</td><td>2.81v0.49</td><td>267.2</td><td>0.81</td><td>0.56</td></tr><tr><td>HART-d20 [69]</td><td>649M</td><td>2.39</td><td>316.4</td><td></td><td></td></tr><tr><td>MVAR-d20 [88]</td><td>600M</td><td>2.87</td><td></td><td>295.30.86</td><td>0.52</td></tr><tr><td>M-VAR-d20 [59]</td><td>900M</td><td>2.41</td><td>308.4</td><td>0.85</td><td>0.58</td></tr><tr><td>HMAR-d20 [31]</td><td>840M</td><td>2.50</td><td>319.0</td><td>0.85</td><td>0.57</td></tr><tr><td>VAR-d20 [73]</td><td>600M</td><td>2.57</td><td>302.6</td><td>0.83</td><td>0.56</td></tr><tr><td>+ Refiner (Ours)</td><td>671M</td><td></td><td>2.17v0.40 274.7</td><td>0.80</td><td>0.60</td></tr><tr><td>HART-d24 [69]</td><td>1.0B</td><td>2.00</td><td>331.5</td><td></td><td></td></tr><tr><td>FastVAR-d24 [23]</td><td>1.0B</td><td>2.64</td><td>287.4</td><td>0.80</td><td>0.58</td></tr><tr><td>MVAR-d24 [88]</td><td>1.0B</td><td>2.23</td><td>300.1</td><td>0.86</td><td>0.52</td></tr><tr><td>M-VAR-d24 [59]</td><td>1.5B</td><td>1.93</td><td>320.7</td><td>0.83</td><td>0.59</td></tr><tr><td>HMAR [31]</td><td>1.3B</td><td>2.10</td><td>319.0</td><td>0.83</td><td>0.60</td></tr><tr><td>VAR-d24 [73]</td><td>1.0B</td><td>2.09</td><td>312.9</td><td>0.83</td><td>0.57</td></tr><tr><td>+ Refiner (Ours)</td><td>1.1B</td><td>1.83v0.26</td><td>288.2</td><td>0.79</td><td>0.63</td></tr><tr><td>HART-d30 [69]</td><td>2.0B</td><td>1.77</td><td>330.3</td><td></td><td></td></tr><tr><td>FastVAR-d30 [23]</td><td>2.0B</td><td>2.30</td><td>288.7</td><td>0.81</td><td>0.59</td></tr><tr><td>VAR-CoDe-d30 [13]</td><td>2.3B</td><td>1.94</td><td>296</td><td>0.81</td><td>0.60</td></tr><tr><td>HMAR [31]</td><td>2.4B</td><td>1.95</td><td>334.5</td><td>0.82</td><td>0.62</td></tr><tr><td>VAR-d30 [73]</td><td>2.0B</td><td>1.92</td><td>323.1</td><td>0.82</td><td>0.58</td></tr><tr><td>+ Refiner (Ours)</td><td>2.2B</td><td></td><td>1.76v0.16 319.4</td><td>0.80</td><td>0.62</td></tr><tr><td>M-VAR-d32 [59]</td><td>3.0B</td><td>1.78</td><td>331.2</td><td>0.83</td><td>0.61</td></tr><tr><td colspan="6">Generative Adversarial Nets [20]</td></tr><tr><td>BigGAN-deep [6]</td><td>112M</td><td>6.95</td><td>202.6</td><td>0.87</td><td>0.28</td></tr><tr><td>StyleGAN-XL [63]</td><td>166M</td><td>2.30</td><td>265.1</td><td>0.78</td><td>0.53</td></tr><tr><td colspan="6">Diffusion [26, 64]</td></tr><tr><td>ADM-G [17]</td><td>554M</td><td>4.59</td><td>186.7</td><td>0.82</td><td>0.52</td></tr><tr><td>LDM-4-G [60]</td><td>400M</td><td>3.60</td><td>247.6</td><td></td><td></td></tr><tr><td>DiT-XL/2 [51]</td><td>675M</td><td>2.27</td><td>278.2</td><td>0.83</td><td>0.57</td></tr><tr><td>SiT-XL/2 [43]</td><td>675M</td><td>2.06</td><td>252.2</td><td></td><td></td></tr><tr><td>JiT-G/16 [36]</td><td>2.0B</td><td>1.82</td><td>292.6</td><td>0.79</td><td>0.62</td></tr><tr><td>FlowAR [57]</td><td>1.9B</td><td>1.65</td><td>296.5</td><td>0.83</td><td>0.60</td></tr><tr><td colspan="6">Raster Autoregression</td></tr><tr><td>VQGAN [18]</td><td>1.4B</td><td>15.78</td><td>74.3</td><td></td><td></td></tr><tr><td>RQ-Transformer [33]</td><td>3.8B</td><td>7.55</td><td>134.0</td><td>0.73</td><td>0.58</td></tr><tr><td>LlamaGen-3B [65]</td><td>3.1B</td><td>2.18</td><td>263.3</td><td>0.81</td><td>0.58</td></tr><tr><td>RAR-XXL [87]</td><td>1.5B</td><td>1.48</td><td>326.0</td><td>0.80</td><td>0.63</td></tr><tr><td>Masked Autoregression</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAGE [35]</td><td>439M</td><td>7.04</td><td>123.5</td><td></td><td></td></tr><tr><td>MaskGiT [8]</td><td>227M 943M</td><td>6.18 1.55</td><td>182.1</td><td>0.80</td><td>0.51</td></tr><tr><td>MAR-H [37]</td><td></td><td></td><td>303.7</td><td>0.81</td><td>0.62</td></tr><tr><td>ImageNet Validation</td><td></td><td>1.78</td><td></td><td></td><td></td></tr></table>

Table B.5: Extra Evaluations without CFG across Scales. Similar to the typical inference setting with CFG (a), our refiner also enables significant gains in generative quality without it (b).
<table><tr><td rowspan="2">FID (↓)</td><td colspan="4">(a) CFG: √</td><td colspan="4">(b) CFG: X</td></tr><tr><td>d16</td><td>d20</td><td>d24</td><td>d30</td><td>d16</td><td>d20</td><td>d24</td><td>d30</td></tr><tr><td>VAR</td><td>3.30</td><td>2.57</td><td>2.09</td><td>1.92</td><td>3.44</td><td>2.62</td><td>2.13</td><td>2.17</td></tr><tr><td>+ Refiner (Ours) 2.810.49 2.170.401.83v0.261.76v0.16</td><td></td><td></td><td></td><td></td><td></td><td>3.410.032.40v0.221.940.191.880.29</td><td></td><td></td></tr></table>

## C Additional Qualitative Samples

## C.1 Text-to-Image

We show additional comparisons between the base Infinity-2B [24] and our refiner version in Supp. Figs. C.2 and C.3.

## C.2 ImageNet

Supp. Figure C.4 shows additional results of VAR with and without our refiner across diferent base model scales. Supp. Figures C.5 to C.10 show uncurated ImageNet samples for various classes.

![](images/155f76a811f7d7cf7fe030b5d0e4bcc002cb34b53278dd2d7a7c0cd5a5976604.jpg)  
The image showcases a detailed and enchanting illustration of a young woman, possibly a sketch in progress, set against a light teal circular backdrop ...

![](images/88acdeb9b451e98afcd40391d37064fd11a4f0c393c6c87e40973a4085195b25.jpg)

![](images/2ece3f69cd68933e6dec6d175b03a3a1f65c10648972d883ee3c2c6ceab2c643.jpg)  
The image features a young woman with a striking blend of modern and traditional Japanese-inspired aesthetics. She is captured from the chest up, with her gaze slightly averted to the right. ...

![](images/1b98672d72c4cd4a0569286cd514ebe7726f39922ccf928105880585c07d1e47.jpg)

![](images/4c63919a58b3ca626bb4593a84a04347a97eda3964bcd4da39bc9c3ca542c2b0.jpg)  
world of warcraft Malygos, the blue dragon aspect, cute tee shirt design illustration, 4k  
Fig. C.2: Selected Infinity-2B + refiner samples on HPSv3 [44] prompts.

![](images/983bb555f0c55f9acdf728c21ebb3c28100395d60fa4d837b3a6cd972872f570.jpg)  
diamond trophy, 2d drawing

![](images/3aaee9ace7773fbcae27b12d242d785acde97ed5d2f103110c8ccb61b118101d.jpg)

![](images/77c1d568e33e5e0c0ed52e26145c1c62417755317d296c114bb7e11af28c9c1a.jpg)  
The whole universe enclosed in a glass globe, exquisite detail.

![](images/dbb67dc3babe3df7b7c6fd9b8d319f48f5364a7f7341291cf3bfa4f9612a43d8.jpg)

![](images/c9558f0191ce91bdb08e745fae070fc9fb1d54f6a4d1c011df2dfe4cdce4839a.jpg)  
a cylindrical 25th century warp core in the style of "Star Trek"  
Fig. C.3: Selected Infinity-2B + refiner samples on HPSv3 [44] prompts.

d16 → d20 → d24 → d30

![](images/cca2080b7566d2c8a941d90c5286513407582ba147ebe24b8c9494dac498eeda.jpg)  
Fig. C.4: Additional Selected Samples from VAR w/o refiner across diferent scales. We use a classifier-free guidance scale 2.5 with topk=500 for sampling across the scales.

![](images/0d5b3fe9b38908815056f65900e760b4c11d191c416975fb2cc5538b604fcf51.jpg)  
(a) Class: “macaw” (88)

![](images/1c2ec3a285497ea320a33ec7889147a05b86860cadc0d45a4c31b5ca25a746df.jpg)  
(b) Class: “Sulphur-crested cockatoo” (89)  
Fig. C.5: Uncurated 256×256 VARd30 + refiner samples. Classifier-free guidance scale = 2.5, topk = 500.

![](images/6258e2dc6a220d5adc30b50fe71141cba45d47a199dcec30aeab3e36cbd283a4.jpg)  
(a) Class: “alp” (980)

![](images/f74eabd86e5d9134152026241ebcc85d2715c84bf26981796b8334f6cc281d4c.jpg)  
(b) Class: “volcano” (970)  
Fig. C.6: Uncurated 256×256 VARd30 + refiner samples. Classifier-free guidance scale = 2.5, topk = 500.

![](images/aba8502cc15a9c5a6840f1c835df5f8e721856948ce982b3301b098f324c5413.jpg)  
(a) Class: “brown bear” (294)

![](images/830a3e01b3957acdadce2815de4f2e64b95da4c4e0261de9247a945931301b8d.jpg)  
(b) Class: “giant panda” (388)  
Fig. C.7: Uncurated 256×256 VARd30 + refiner samples. Classifier-free guidance scale = 2.5, topk = 500.

![](images/016ef31b72861a8922fa2479dc1e6e9668c7fd62b875f0e3103cb8e7cb24bff9.jpg)  
(a) Class: “triumphal arch” (873)

![](images/93ff7f5109930cae6927f1354fc67dac85b986d545d582ea2c2c2da9ebf2281b.jpg)  
(b) Class: “totem pole” (863)  
Fig. C.8: Uncurated 256×256 VARd30 + refiner samples. Classifier-free guidance scale = 2.5, topk = 500.

![](images/ba5d636a2bad8bd6bfd11ea64ebf546b1de84b15fbfd14ce90e322bed49f715e.jpg)  
(a) Class: “tabby” (281)

![](images/fd78a8e86dfdfad612fe7736ee9c1167a649594ce8abc7f139c95e699c44fe60.jpg)  
(b) Class: “golden retriever” (207)  
Fig. C.9: Uncurated 256×256 VARd30 + refiner samples. Classifier-free guidance scale = 2.5, topk = 500.

![](images/3eeb89c544f25ec867e0615fc3c17574924af2346ff4cb6286ddbee1f7f530f0.jpg)  
(a) Class: “electric locomotive” (547)

![](images/049e1226b573dc8ff4151d314d0c842c070fcf43d096866cf99784ef2f1fd4cd.jpg)  
(b) Class: “convertible” (511)  
Fig. C.10: Uncurated 256×256 VARd30 + refiner samples. Classifier-free guidance scale = 2.5, topk = 500.