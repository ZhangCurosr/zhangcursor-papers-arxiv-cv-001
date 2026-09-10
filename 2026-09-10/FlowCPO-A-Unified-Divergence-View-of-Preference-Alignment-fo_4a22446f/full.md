# FlowCPO: A Unified Divergence View of Preference Alignment for Flow Models

Yansen Han<sup>1,2,∗</sup> Shengyi Liao<sup>3,∗</sup> Peng Sun<sup>1,2</sup> Deyuan Liu<sup>1</sup> Yuanxing Zhang<sup>3</sup>

Pengfei Wan<sup>3</sup> Tao Lin<sup>1,†</sup>

<sup>∗</sup>Equal contribution <sup>†</sup>Corresponding author

<sup>1</sup>Westlake University <sup>2</sup>Zhejiang University <sup>3</sup>Kling Team, Kuaishou Technology

## Abstract

Preference alignment for flow and diffusion models now spans online reinforcement learning and offline preference optimization, but the relation between these methods remains unclear. In particular, existing forward-process alignment methods require fresh samples from the current model, while offline methods based on fixed preference pairs rely primarily on positive-only fine-tuning or DPO-style likelihood-ratio surrogates. We organize these approaches through a divergencebased framework and introduce FLOWCPO, an offline forward-KL objective that uses both preferred and dispreferred samples without online rollouts. For linear interpolation, we show under explicit regularity conditions that the forward-KL objective is bounded by a contrastive flow matching loss, yielding a tractable surrogate on fixed data. We further show that this loss is nonnegative, whereas the signed regression loss of simplified FlowDPO can be unbounded below. In the in-domain setting, FLOWCPO achieves higher mean GenEval and OCR scores than the evaluated baselines, reaching 0.84 and 0.87 versus 0.81 and 0.74 for FlowDPO at CFG 3.0. In the out-of-domain setting, the results are mixed, with the best GenEval result but lower reward scores than RFT on several metrics.

## 1 Introduction

Recent advancements in flow matching and diffusion-based generative modeling [19, 30, 1, 40, 35, 20] have facilitated the development of practical preference alignment algorithms for continuous generative spaces [32, 55, 33, 51, 3]. These efforts have yielded diverse alignment paradigms, ranging from online reinforcement learning frameworks like FlowGRPO [32], DiffusionNFT [55], AWM [51] and RAM [3] to offline preference optimization methods such as FlowDPO [33]. In contrast to alignment in discrete autoregressive models, these continuous-time methods operate directly on sampling trajectories and velocity fields rather than token-level probabilities. While empirical results are promising, the relations among these methods remain difficult to compare.

Existing flow-based alignment methods can be organized along two axes (Tab. 1): the sampling regime (online versus offline) and the direction of the idealized KL objective. Under this view, FlowGRPO [32] and FlowDPO [33] represent online and offline instances of the reverse-KL branch, whereas DiffusionNFT [55] provides a forward-process objective but refreshes its train-

Table 1: Objective-level view of flow-based preference alignment. The two axes separate the sampling regime from the direction of the idealized KL objective.
<table><tr><td>Divergence</td><td>Online</td><td>Offline</td></tr><tr><td>Reverse-KL</td><td>FlowGRPO</td><td>FlowDPO</td></tr><tr><td>Forward-KL</td><td>DiffusionNFT</td><td>FLOWCPO (ours)</td></tr></table>

ing samples online. The unresolved problem is therefore the offline forward-KL objective using both preferred and dispreferred target samples: a forward-KL expectation can use fixed target samples, but its log-likelihood is intractable for flow matching. An offline method needs a justified bridge from this distribution-level objective to velocity-field regression on fixed preferred and dispreferred data.

![](images/50f89b1795d63e87116dba4e40a5992719200d09a42c0050618d7d2fd6a93d53.jpg)  
Figure 1: FLOWCPO uses a fixed preference dataset to train two coupled flow branches. A frozen generator supplies preferred/dispreferred pairs D = {(c, x<sup>w</sup><sub>0</sub> , x<sup>l</sup><sub>0</sub>)}; the positive branch matches preferred samples, while a mirrored branch matches dispreferred samples. Training uses only noised versions of these fixed endpoints and requires no online rollout.

We address this gap by establishing a unified divergence-based framework for flow-based preference alignment and deriving Flow Contrastive Preference Optimization (FLOWCPO). Starting from forward-KL terms over preferred and dispreferred target distributions, we use a flow matching bound for linear interpolation to obtain a contrastive regression surrogate. The resulting loss can be estimated from a fixed preference dataset and therefore requires no sampling from the model during training.

We also connect FLOWCPO to simplified FlowDPO (App. B.5). Simplified FlowDPO subtracts the regression error on dispreferred samples and can be unbounded below, whereas FLOWCPO adds two nonnegative regression errors. With equal branch weights (λ = 1), our loss decomposition identifies corrections determined by the difference between the current model and its exponential moving average (EMA).

Empirically, we evaluate our approach under a strictly offline protocol. In the in-domain setting, where preference pairs are generated by the reference model, FLOWCPO attains higher mean GenEval and OCR scores than RFT and FlowDPO while remaining strong on general-preference metrics. In the out-of-domain setting, the result is more mixed: FLOWCPO gives the best GenEval score but RFT is stronger on several reward metrics. The overall offline optimization pipeline is illustrated in Fig. 1. Our contributions are threefold:

• We provide a unified divergence-based framework that separates the online/offline sampling regime from the direction of the idealized KL objective, clarifying the assumptions behind connections among existing flow-based alignment methods.

• We derive FLOWCPO, a nonnegative contrastive surrogate for offline forward-KL alignment. Under stated regularity conditions for linear interpolation, its regression objective bounds the branch-wise forward-KL objective and is estimable from fixed preference data. We also derive its loss decomposition relative to simplified FlowDPO.

• We evaluate FLOWCPO with two types of offline data, showing higher GenEval and OCR scores than FlowDPO in the in-domain setting and mixed results in the out-of-domain setting.

## 2 Related Work

Online alignment with policy-generated samples. Online methods improve a generator using samples drawn from its current or recent policy. ReFL [50] uses reward feedback, DDPO [4] and DPOK [10] estimate trajectory policy gradients, and DRaFT [6] and AlignProp [37] differentiate through the sampler. FlowGRPO [32] optimizes stochasticized flow trajectories, with subsequent refinements to temporal allocation and trajectory reuse [16, 27, 28, 8]. SPO [29] generates candidates at each denoising step and selects preference pairs with a step-aware model. DiffusionNFT [55], AWM [51], and RAM [3] use regression-style objectives on noised on-policy samples.

Offline alignment with fixed preference data. Fixed-data methods differ in how they use preferred and dispreferred samples. D3PO [52] extends pairwise learning to diffusion trajectories;

DiffusionDPO [45] uses reference-relative denoising errors, with temporal weighting in Dense Reward [53] and a flow-model extension in FlowDPO [33]. DMPO [26] develops a reverse-KL objective, whereas MaPO [21] learns preference margins without a reference model. RFT [49, 5] fits only preferred samples. FLOWCPO instead uses both sides of fixed preference pairs in a coupled, two-branch forward-KL objective, with a flow-matching upper-bound surrogate under stated conditions.

Divergence-based alignment in language models. LLM alignment also treats divergence and utility as design choices. f-DPG [13] generalizes distribution matching, while f-DPO [46] varies the divergence constraint for preference learning. GPO [44] relates DPO [38], IPO [2], and SLiC [54] through convex pairwise losses. f-PO [14] unifies divergence-minimization formulations including DPO and EXO [22].

## 3 Preliminaries

We collect notation and background needed for our divergence-based view of flow alignment. Specifically, Sec. 3.1 recalls diffusion models and flow matching under linear interpolation. Sec. 3.2 summarizes RLHF and DPO in the discrete policy formulation, which we then lift to continuous-time models in Sec. 3.3. Finally, Sec. 3.4 reviews forward and reverse KL divergences.

## 3.1 Diffusion Models and Flow Matching

We first briefly review the two continuous-time generative paradigms used throughout the paper. Both start from a simple latent variable, typically Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ ), and define intermediate states $\mathbf { x } _ { t }$ for $t \in [ 0 , 1 ]$ that connect data and noise.

In diffusion models [19, 40], one specifies a forward noising process $q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } )$ and trains a network $\epsilon _ { \theta } ( \mathbf { x } _ { t } , c , t )$ , or equivalently a score model, to predict the noise added to the clean sample $\mathbf { x } _ { \mathrm { 0 } }$ Generation then approximately reverses this process from noise to data through a reverse-time SDE or the associated probability flow ODE. Flow matching [30, 1] instead directly learns the vector field of a prescribed probability path. In this paper we use the linear interpolation

$$
{ \bf x } _ { t } = ( 1 - t ) \cdot { \bf x } _ { 0 } + t \cdot \epsilon , \qquad \epsilon \sim { \mathcal N } ( 0 , I ) ,\tag{1}
$$

so that $\mathbf { x } _ { \mathrm { 0 } }$ is the clean sample at $t = 0$ and ${ \bf x } _ { 1 } = \epsilon$ is the noise sample at $t = 1$ . The corresponding conditional target velocity is

$$
u _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = { \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } } = \epsilon - \mathbf { x } _ { 0 } .\tag{2}
$$

Flow matching trains a velocity field $v _ { \theta } ( \mathbf { x } _ { t } , c , t )$ to regress to this target, typically by minimizing

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { c , \mathbf { x } _ { 0 } , t , \epsilon } \left[ | | v _ { \theta } ( \mathbf { x } _ { t } , c , t ) - u _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) | | _ { 2 } ^ { 2 } \right] . } \end{array}\tag{3}
$$

Although diffusion and flow matching use different parameterizations, they are closely connected: under suitable probability paths, the denoising model, score function, and velocity field describe the same underlying transport from noise to data [24, 20, 43, 31].

## 3.2 RLHF and Direct Preference Optimization (DPO)

Reinforcement Learning from Human Feedback (RLHF) [36] typically aligns a generative policy $\pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 0 } | c )$ by maximizing expected reward while regularizing deviation from a reference policy, where $\mathbf { x } _ { \mathrm { 0 } }$ is the generated content and c is the condition. We use $r ( \mathbf { x } _ { 0 } , c )$ to denote the reward function, $\mathbf { x } _ { 0 } ^ { w }$ and $\mathbf { x } _ { 0 } ^ { l }$ to denote the preferred and dispreferred contents, respectively. DPO [38, 42, 44] bypasses the explicit reward modeling step by directly optimizing the policy using a closed-form solution to the KL-constrained reward maximization problem. Given a dataset $\boldsymbol { \mathcal { D } } = \bar { \{ ( c , \mathbf { x } _ { 0 } ^ { w } , \mathbf { x } _ { 0 } ^ { l } ) \} }$ } of preferred $( \mathbf { x } _ { 0 } ^ { w } )$ and dispreferred $( \mathbf { x } _ { 0 } ^ { l } )$ contents, DPO minimizes the following negative log-likelihood loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \theta ) = - \mathbb { E } _ { ( c , \mathbf { x } _ { 0 } ^ { w } , \mathbf { x } _ { 0 } ^ { l } ) \sim \mathcal { D } } \left[ \log \sigma \left( \beta \log \frac { \pi _ { \theta } \left( \mathbf { x } _ { 0 } ^ { w } \mid c \right) } { \pi _ { \mathrm { r e f } } \left( \mathbf { x } _ { 0 } ^ { w } \mid c \right) } - \beta \log \frac { \pi _ { \theta } \left( \mathbf { x } _ { 0 } ^ { l } \mid c \right) } { \pi _ { \mathrm { r e f } } \left( \mathbf { x } _ { 0 } ^ { l } \mid c \right) } \right) \right] , } \end{array}\tag{4}
$$

where $\pi _ { \mathrm { r e f } }$ is the frozen reference policy and $\beta$ is a temperature parameter controlling the strength of the KL constraint.

## 3.3 Preference Optimization in Continuous-Time Models

Applying equation 4 directly to the diffusion and flow-matching models introduced above is non-trivial. Calculating the exact log-likelihood log $\pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 0 } | c )$ requires solving the probability flow ODE, which is computationally prohibitive during training [55]. DiffusionDPO [45] addresses this intractability by optimizing the Evidence Lower Bound (ELBO) as a proxy for the likelihood. The loss function is reformulated using the denoising error at timestep t:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D i f f - D P O } } ( \theta ) = - \mathbb { E } _ { c , \mathbf { x } _ { 0 } ^ { w } , \mathbf { x } _ { 0 } ^ { l } , t , \epsilon } \Big [ \log \sigma \Big ( \beta \rho _ { t } \Big ( } \\ & { \qquad \underbrace { ( \lVert \epsilon _ { \theta } ( \mathbf { x } _ { t } ^ { l } ) - \epsilon \rVert ^ { 2 } - \lVert \epsilon _ { \mathrm { r e f } } ( \mathbf { x } _ { t } ^ { l } ) - \epsilon \rVert ^ { 2 } ) } _ { \mathrm { A d v a r a g e ~ o f ~ L o s e r } } - \underbrace { \left( \lVert \epsilon _ { \theta } ( \mathbf { x } _ { t } ^ { w } ) - \epsilon \rVert ^ { 2 } - \lVert \epsilon _ { \mathrm { r e f } } ( \mathbf { x } _ { t } ^ { w } ) - \epsilon \rVert ^ { 2 } \right) } _ { \mathrm { A d v a r a g e ~ o f ~ W i n n e r } } \Big ) \Big ) \Big ] } \end{array}\tag{5}
$$

where $\mathbf { x } _ { t }$ is the linear interpolation of $\mathbf { x } _ { \mathrm { 0 } }$ and noise $\epsilon ,$ and $\rho _ { t }$ is a weighting function. FlowDPO [33] extends this paradigm to flow matching. By leveraging the connection between the score function and the vector field, FlowDPO replaces the noise prediction error with the flow matching loss, enabling efficient gradient-based preference optimization directly on the velocity field without ODE integration.

## 3.4 Forward and Reverse KL Divergences

Let $\pi ( \mathbf { x } _ { 0 } \mid c )$ denote a target distribution and $\pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 0 } \mid c )$ denote the trainable model distribution. Later in the paper, π will be instantiated by the preferred and dispreferred target distributions $\pi ^ { + }$ and $\pi ^ { - }$ , while the model side of the objective will be represented by branch-specific distributions $q _ { \theta } ^ { + }$ and $q _ { \theta } ^ { - }$ rather than by a single $\pi _ { \theta }$ . The two KL directions are

$$
\begin{array} { r } { \mathrm { R e v e r s e - K L : } \mathcal { D } _ { K L } ( \pi _ { \theta } | | \pi ) = \mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot | c ) } \left[ \log \frac { \pi _ { \theta } \left( \mathbf { x } _ { 0 } \mid c \right) } { \pi \left( \mathbf { x } _ { 0 } \mid c \right) } \right] } \\ { \mathrm { F o r w a r d - K L : } \mathcal { D } _ { K L } ( \pi | | \pi _ { \theta } ) = \mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi ( \cdot | c ) } \left[ \log \frac { \pi \left( \mathbf { x } _ { 0 } \mid c \right) } { \pi _ { \theta } \left( \mathbf { x } _ { 0 } \mid c \right) } \right] } \end{array}\tag{6}
$$

The key difference is the sampling distribution inside the expectation. In the reverse-KL term ${ \mathcal { D } } _ { K L } ( \pi _ { \theta } \| \pi )$ , errors are weighted by the current model $\pi _ { \boldsymbol { \theta } } .$ , so target regions that are rarely visited by π<sub>θ</sub> contribute little. In contrast, the forward-KL term ${ \mathcal { D } } _ { K L } ( \pi \| \pi _ { \theta } )$ is weighted by the target distribution π. If $\pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 0 } \mid c )$ is too small in regions where $\pi ( \mathbf { x } _ { 0 } \mid c )$ is large, the penalty becomes severe, and finiteness requires $\operatorname { s u p p } ( \pi ) \subseteq \operatorname { s u p p } ( \pi _ { \theta } )$

## 4 Methodology

This section develops FLOWCPO from a unified divergence-based formulation of preference alignment. We first show that RLHF admits a contrastive reverse-KL interpretation, then derive its offline forward-KL counterpart and reduce the resulting objective to a practical flow matching loss.

## 4.1 Probabilistic Reward Formulation

Let c denote the conditioning context and $\mathbf { x } _ { \mathrm { 0 } }$ the generated content. Following control-asinference [25], we introduce a binary optimality variable $o \in \{ 0 , 1 \}$ , where $o = 1$ means $\mathbf { x } _ { \mathrm { 0 } }$ is preferred and $o = 0$ means $\mathbf { x } _ { \mathrm { 0 } }$ is dispreferred. This gives a normalized probabilistic view of any unbounded reward $\mathbf { \nabla } r ( \mathbf { x } _ { 0 } , c ) \mathbf { ; }$

$$
\begin{array} { l } { p ( o = 1 \mid \mathbf { x } _ { 0 } , c ) = \frac { \exp ( \omega \cdot r ( \mathbf { x } _ { 0 } , c ) ) } { \exp ( \omega \cdot r ( \mathbf { x } _ { 0 } , c ) ) + \exp ( - \omega \cdot r ( \mathbf { x } _ { 0 } , c ) ) } = \sigma ( 2 \omega \cdot r ( \mathbf { x } _ { 0 } , c ) ) } \\ { p ( o = 0 \mid \mathbf { x } _ { 0 } , c ) = \frac { \exp ( - \omega \cdot r ( \mathbf { x } _ { 0 } , c ) ) } { \exp ( \omega \cdot r ( \mathbf { x } _ { 0 } , c ) ) + \exp ( - \omega \cdot r ( \mathbf { x } _ { 0 } , c ) ) } = \sigma ( - 2 \omega \cdot r ( \mathbf { x } _ { 0 } , c ) ) } \end{array}\tag{7}
$$

Here $\omega > 0$ is an inverse temperature and $\begin{array} { r } { \sigma ( x ) = \frac { 1 } { 1 + \exp ( - x ) } } \end{array}$ is the sigmoid function. Conditioning the reference policy $\pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } \mid c )$ on the two optimality events, $o = 1$ and $o = 0$ , yields preferred and dispreferred target distributions:

$$
\pi ^ { + } ( \mathbf { x } _ { 0 } \mid c ) : = p ( \mathbf { x } _ { 0 } \mid o = 1 , c ) = { \frac { \pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } \mid c ) p ( o = 1 \mid \mathbf { x } _ { 0 } , c ) } { p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) } }
$$

$$
\pi ^ { - } ( \mathbf { x _ { 0 } } \mid c ) : = p ( \mathbf { x _ { 0 } } \mid o = 0 , c ) = { \frac { \pi _ { \mathrm { r e f } } ( \mathbf { x _ { 0 } } \mid c ) p ( o = 0 \mid \mathbf { x _ { 0 } } , c ) } { p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) } }\tag{8}
$$

We use $\pi ^ { + }$ and $\pi ^ { - }$ as the positive and negative target distributions throughout the paper. In practice, offline preference construction may only produce empirical approximations to these posteriors. We defer this distinction to App. B.1.

By inverting equation 8 and substituting into equation 7, the reward $r ( \mathbf { x } _ { 0 } , c )$ becomes

$$
r ( \mathbf { x } _ { 0 } , c ) = { \frac { 1 } { 2 \omega } } \log { \frac { p ( o = 1 \mid \mathbf { x } _ { 0 } , c ) } { p ( o = 0 \mid \mathbf { x } _ { 0 } , c ) } } = { \frac { 1 } { 2 \omega } } \left[ \log { \frac { \pi ^ { + } ( \mathbf { x } _ { 0 } \mid c ) } { \pi ^ { - } ( \mathbf { x } _ { 0 } \mid c ) } } + \log { \frac { p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) } { p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) } } \right] .\tag{9}
$$

Thus, up to a context-only constant, reward is a contrastive log-density ratio: high reward corresponds to regions where $\pi ^ { + }$ dominates $\pi ^ { - }$

## 4.2 Re-formalizing Reinforcement Learning from Human Feedback

We begin with the unregularized expected-reward term $\mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( . | c ) } [ r ( \mathbf { x } _ { 0 } , c ) ]$ that appears inside RLHF objectives. Using equation 9, we can rewrite this term directly in divergence form in Thm. 4.1 (see App. B.2 for the proof):

Theorem 4.1 (Expected reward as contrastive reverse KL). Up to the positive scale factor $\frac { 1 } { 2 \omega }$ and an additive term depending only on c, maximizing expected reward is equivalent to minimizing

$$
\mathbb { E } _ { c } \left[ \mathcal { D } _ { K L } ( \pi _ { \theta } \| \pi ^ { + } ) - \mathcal { D } _ { K L } ( \pi _ { \theta } \| \pi ^ { - } ) \right] .\tag{10}
$$

Equation 10 shows that expected reward induces a contrastive reverse-KL term: the policy is attracted to $\pi ^ { + }$ and repelled from $\pi ^ { - }$ under its own sampling distribution. Reference-policy KL penalties used in practical RLHF methods remain additional terms and are not absorbed by this identity. Due to the high computational cost of online sampling, we focus on the offline regime in this work. To derive an offline counterpart, we reverse the direction of the KL divergence in equation 10 and assume fixed sampling distributions:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { O f f i n e } } = \mathbb { E } _ { c } \left[ \mathcal { D } _ { K L } ( \pi ^ { + } \| \pi _ { \theta } ^ { + } ) + \lambda \cdot \mathcal { D } _ { K L } ( \pi ^ { - } \| \pi _ { \theta } ^ { - } ) \right] , \qquad \lambda \ge 0 .\tag{11}
$$

Here $\pi _ { \theta } ^ { + }$ and $\pi _ { \theta } ^ { - }$ are branch-specific model distributions. For FLOWCPO itself, we focus on the attractive regime $\lambda \geq 0$ . In Sec. 5.3, we additionally instantiate $\lambda < 0$ under the generalized framework to analyze repulsive baselines.

To avoid evaluating flow-model likelihoods directly, we parameterize the two branches through symmetric mixed velocity fields:

$$
\mu _ { \theta } = \left( 1 - \beta \right) v _ { \mathrm { o l d } } + \beta v _ { \theta } , \qquad \nu _ { \theta } = \left( 1 + \beta \right) v _ { \mathrm { o l d } } - \beta v _ { \theta } .
$$

Here $v _ { \mathrm { o l d } }$ is an EMA copy of the trainable field, and $\beta > 0$ controls the symmetric displacement of the two branches from that reference. The implicit distributions induced from the common Gaussian endpoint by $\mu _ { \theta }$ and $\nu _ { \theta }$ are denoted by $\pi _ { \theta } ^ { + }$ and $\pi _ { \theta } ^ { - }$ . The idealized score-field interpretation of this interpolation is given in App. B.3.

The remaining bridge is from cross-entropy to regression. Each forward-KL term equals a target entropy plus an expected model negative log-likelihood; under the uniform regularity conditions in App. A, the latter is bounded by conditional flow matching objective plus a constant independent of θ. Applying this argument separately to the two branches yields Thm. 4.2.

Theorem 4.2 (A flow-matching upper bound for offline forward KL). Under the regularity conditions in App. A, we have:

$$
\begin{array} { r l } & { \mathcal { L } _ { O f l i n e } ( \theta ) \leq C + \mathbb { E } _ { c , \mathbf { x } _ { 0 } ^ { + } \sim \pi ^ { + } ( \cdot \vert c ) } \Big [ \big \| \mu _ { \theta } ( \mathbf { x } _ { t } ^ { + } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { + } \mid \mathbf { x } _ { 0 } ^ { + } ) \big \| _ { 2 } ^ { 2 } \Big ] } \\ & { \qquad + \lambda \cdot \mathbb { E } _ { c , \mathbf { x } _ { 0 } ^ { - } \sim \pi ^ { - } ( \cdot \vert c ) } \Big [ \big \| \nu _ { \theta } ( \mathbf { x } _ { t } ^ { - } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { - } \mid \mathbf { x } _ { 0 } ^ { - } ) \big \| _ { 2 } ^ { 2 } \Big ] , } \end{array}\tag{12}
$$

where C is independent ofθ.

The proof of Thm. 4.2 is deferred to App. B.4. This result justifies a tractable upper-bound surrogate. In practice, we replace the idealized source distributions $\pi ^ { + } , \pi ^ { - }$ with empirical preferred and dispreferred datasets, $\mathcal { D } ^ { + }$ and $\mathcal { D } ^ { - }$ , and minimize the following contrastive flow matching loss:

Table 2: Comparison of flow-based alignment methods and related special cases under equation 14. Rows are organized by optimization regime and divergence choice. BT indicates Bradley–Terry preference model, and Ref. KL indicates whether the method uses reference-model KL regularization.
<table><tr><td>Method</td><td>Regime</td><td>Divergence</td><td>Optimized Process</td><td>α</td><td>γ</td><td> ${ \boldsymbol q } _ { \theta } ^ { + }$ </td><td> $q _ { \theta } ^ { - }$ </td><td>BT</td><td>Ref. KL</td></tr><tr><td>FlowGRPO [32]</td><td>Online</td><td>Reverse KL</td><td>Reverse Process</td><td>1</td><td>-1</td><td>πθ</td><td>πθ</td><td>X</td><td>√</td></tr><tr><td>DiffusionNFT [55]</td><td>Online</td><td>Forward KL</td><td>Forward Process</td><td> $p ( o = 1 | c )$ </td><td> $p ( o = 0 | c )$ </td><td>π</td><td>πθ</td><td>X</td><td>×</td></tr><tr><td>AWM [51]</td><td>Online</td><td>Reverse KL</td><td>Forward Process</td><td>1</td><td>-1</td><td>πθ</td><td>πθ</td><td>×</td><td>×</td></tr><tr><td>RAM [3]</td><td>Online</td><td>Reverse KL</td><td>Forward Process</td><td>1</td><td>-1</td><td>πθ</td><td>πθ</td><td>×</td><td>√</td></tr><tr><td>SFT</td><td>Offline</td><td>Forward KL</td><td>-</td><td> $p ( o = 1 | c )$ </td><td> $p ( o = 0 | c )$ </td><td>πθ</td><td>πθ</td><td>×</td><td>×</td></tr><tr><td>RFT [49, 5]</td><td>Offline</td><td>Forward KL</td><td></td><td>1</td><td>0</td><td>πθ</td><td>一</td><td>X</td><td>×</td></tr><tr><td>FlowDPO [33]</td><td>Offline</td><td>Implicit reverse KL</td><td></td><td>1</td><td>-1</td><td>πθ</td><td>πθ</td><td>√</td><td>√</td></tr><tr><td>FLOWCPO</td><td>Offline</td><td>Forward KL</td><td></td><td>1</td><td>λ</td><td>π</td><td> $\pi _ { \theta } ^ { - }$ </td><td>×</td><td>×</td></tr></table>

$$
\begin{array} { r l } { \underset { \theta } { \operatorname* { m i n } } \mathcal { L } _ { \mathrm { F L o w C P o } } ( \theta ) = } & { \quad \mathbb { E } _ { ( c , \mathbf { x } _ { 0 } ^ { w } ) \sim \mathcal { D } ^ { + } } [ \| \mu _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } ] } \\ & { \quad +  \lambda \cdot \mathbb { E } _ { ( c , \mathbf { x } _ { 0 } ^ { l } ) \sim \mathcal { D } ^ { - } } [ \| \nu _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } ] . } \end{array}\tag{13}
$$

Pseudo-code is given in Alg. 1. For $\lambda \geq 0 .$ , equation 13 is bounded below by zero, whereas simplified FlowDPO’s signed objective can be unbounded below. App. B.5 provides the comparison.

## 4.3 Unified View and Connections to Prior Work

Tab. 2 summarizes connections between equation 14 and prior works, and App. C elaborates the assumptions behind each mapping. FLOWCPO is derived from the forward-KL objective in equation 11, which is a special case of equation 14. The generalized objective organizes flow-based preference-alignment methods (FlowDPO, FlowGRPO, DiffusionNFT) and regression baselines (SFT, RFT):

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { G e n e r a l } } = \mathbb { E } _ { c } \left[ \alpha \cdot \underbrace { \mathcal { D } _ { f } \big ( \pi ^ { + } \| q _ { \theta } ^ { + } \big ) } _ { \mathrm { p o s i t i v e ~ a l i g n m e n t } } + \gamma \cdot \underbrace { \mathcal { D } _ { f } \big ( \pi ^ { - } \| q _ { \theta } ^ { - } \big ) } _ { \mathrm { n e g a t i v e ~ r e g u l a r i z a t i o n } } \right]\tag{14}
$$

Here $\alpha > 0$ , and $\gamma \in \mathbb { R }$ . In the forward-KL branch we use $\mathcal { D } _ { f } ( \pi ^ { \pm } \| q ) = \mathcal { D } _ { K L } ( \pi ^ { \pm } \| q )$ , and in the reverse-KL branch we use $\mathcal { D } _ { f } ( \pi ^ { \pm } \| q ) = \mathcal { D } _ { K L } ( q \| \pi ^ { \pm } )$ . When $\gamma \geq 0 ,$ the negative regularization matches a negative distribution, and it acts as repulsive regularization when $\gamma < 0$

## 5 Experiments

We study FLOWCPO along three axes: offline alignment with in-domain and out-of-domain preference data, the effect of negative regularization, and sensitivity to the core hyperparameters $( \bar { \beta , } \lambda , \eta )$ The main text focuses on the central quantitative and qualitative results, while the appendix collects the training pseudo-code, per-domain tables, and extended ablations.

## 5.1 Experimental Setup

Data Construction and Implementation. We evaluate FLOWCPO on Stable Diffusion 3.5 Medium (SD3.5-M) [9] in a strictly offline setting. Our offline preference data come from two sources. In the in-domain regime, we build winner/loser pairs from a frozen copy of the reference model using prompts drawn from the target benchmark sources. In the out-of-domain (OOD) regime, we use the public Open Image Preferences v1 dataset, whose images are generated by other open models. All models are trained under the same offline protocol, and we defer the exact data-generation pipeline, dataset sizes, reward filtering, and training hyperparameters to App. D.1. In the in-domain setting, each target capability uses its own specialist model, so the Tab. 3 summarizes task-specific fine-tuning results rather than a single universal model.

Table 3: Quantitative comparison under in-domain and out-of-domain (OOD) offline training regimes on SD3.5-M. We report results with and without Classifier-Free Guidance (CFG). When CFG is enabled, we report separate rows for guidance scales {1.0, 3.0, 4.5}. Under the in-domain setting, fine-tuned methods are trained separately for each target domain (Concept, Typography, and General Preference). Under the OOD setting, a single model is trained on the OOD offline dataset and evaluated across all domains and metrics. The two blocks therefore correspond to different evaluation protocols and should be compared primarily within, rather than across, regimes. The main table reports means only and rounds all entries to 2-3 decimal places; the corresponding mean ± standard deviation statistics over 5 independent runs for fine-tuned SD3.5-M variants are reported in the App. D.2, whereas pretrained baselines are single evaluations. Within each training regime, the best result is highlighted in bold and the second-best result is underlined.
<table><tr><td></td><td></td><td>Concept</td><td>Typography</td><td colspan="6">General Preference</td></tr><tr><td>Model</td><td>CFG</td><td>GenEval ↑</td><td>OCR↑</td><td>PickScore ↑</td><td>CLIPSc. ↑</td><td>HPSv2.1 ↑</td><td>Aes. ↑</td><td>ImgRwd↑</td><td>UniRwd ↑</td></tr><tr><td colspan="10">Pretrained model baselines</td></tr><tr><td>SD-XL</td><td></td><td>0.55</td><td>0.14</td><td>22.42</td><td>0.287</td><td>0.280</td><td>5.60</td><td>0.76</td><td>2.93</td></tr><tr><td>SD3.5-L</td><td>一</td><td>0.71</td><td>0.68</td><td>22.91</td><td>0.289</td><td>0.288</td><td>5.50</td><td>0.96</td><td>3.25</td></tr><tr><td>FLUX.1-Dev</td><td>一</td><td>0.66</td><td>0.59</td><td>22.84</td><td>0.295</td><td>0.274</td><td>5.71</td><td>0.96</td><td>3.27</td></tr><tr><td colspan="10">Reference model: SD3.5-M</td></tr><tr><td></td><td>1.0</td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td>Base Model</td><td>3.0</td><td>0.59</td><td>0.47</td><td>22.28</td><td>0.287</td><td>0.284</td><td>5.38</td><td>0.71</td><td>2.96</td></tr><tr><td></td><td>4.5</td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td colspan="10">In-domain offline fine-tuning on SD3.5-M</td></tr><tr><td></td><td>1.0</td><td>0.59</td><td>0.35</td><td>21.91</td><td>0.279</td><td>0.276</td><td>5.35</td><td>0.57</td><td>2.66</td></tr><tr><td>+ RFT [49, 5]</td><td>3.0</td><td>0.74</td><td>0.70</td><td>22.60</td><td>0.296</td><td>0.304</td><td>5.41</td><td>1.06</td><td>3.14</td></tr><tr><td></td><td>4.5</td><td>0.75</td><td>0.72</td><td>22.57</td><td>0.297</td><td>0.305</td><td>5.42</td><td>1.11</td><td>3.15</td></tr><tr><td></td><td>1.0</td><td>0.59</td><td>0.51</td><td>20.72</td><td>0.240</td><td>0.221</td><td>5.22</td><td>-0.46</td><td>2.11</td></tr><tr><td>+ FlowDPO [33]</td><td>3.0</td><td>0.81</td><td>0.74</td><td>22.76</td><td>0.297</td><td>0.301</td><td>5.56</td><td>1.04</td><td>3.10</td></tr><tr><td></td><td>4.5</td><td>0.81</td><td>0.75</td><td>22.89</td><td>0.301</td><td>0.311</td><td>5.56</td><td>1.15</td><td>3.18</td></tr><tr><td></td><td>1.0</td><td>0.76</td><td>0.83</td><td>22.48</td><td>0.280</td><td>0.290</td><td>5.64</td><td>0.90</td><td>2.93</td></tr><tr><td>+ FLOWCPO (β = 0.5, Ours)</td><td>3.0</td><td>0.84</td><td>0.87</td><td>22.94</td><td>0.299</td><td>0.311</td><td>5.54</td><td>1.25</td><td>3.30</td></tr><tr><td></td><td>4.5</td><td>0.82</td><td>0.86</td><td>22.67</td><td>0.300</td><td>0.303</td><td>5.46</td><td>1.22</td><td>3.28</td></tr><tr><td colspan="10">Out-of-domain (OOD) offline fine-tuning on SD3.5-M</td></tr><tr><td>+ RFT [49, 5]</td><td>1.0</td><td>0.44</td><td>0.17</td><td>21.59</td><td>0.268</td><td>0.261</td><td>5.49</td><td>0.31</td><td>2.56</td></tr><tr><td></td><td>3.0 4.5</td><td>0.67 0.68</td><td>0.53</td><td>22.64</td><td>0.297 0.299</td><td>0.303 0.307</td><td>5.48 5.47</td><td>1.05</td><td>3.18 3.23</td></tr><tr><td></td><td></td><td></td><td>0.60</td><td>22.69</td><td></td><td></td><td></td><td>1.12</td><td></td></tr><tr><td></td><td>1.0</td><td>0.23</td><td>0.12</td><td>20.83</td><td>0.243</td><td>0.226</td><td>5.72</td><td>-0.39</td><td>2.27</td></tr><tr><td>+ FlowDPO [33]</td><td>3.0</td><td>0.61</td><td>0.48</td><td>22.54</td><td>0.292</td><td>0.292</td><td>5.52</td><td>0.91</td><td>3.10</td></tr><tr><td></td><td>4.5</td><td>0.65</td><td>0.54</td><td>22.62</td><td>0.295</td><td>0.299</td><td>5.49</td><td>0.99</td><td>3.17</td></tr><tr><td></td><td>1.0</td><td>0.57</td><td>0.24</td><td>22.02</td><td>0.281</td><td>0.275</td><td>5.35</td><td>0.63</td><td>2.85</td></tr><tr><td>+ FLOWCPO (β = 0.5, Ours)</td><td>3.0</td><td>0.69</td><td>0.49</td><td>22.32</td><td>0.296</td><td>0.288</td><td>5.37</td><td>1.00</td><td>3.15</td></tr><tr><td></td><td>4.5</td><td>0.68</td><td>0.49</td><td>22.12</td><td>0.294</td><td>0.284</td><td>5.31</td><td>0.95</td><td>3.11</td></tr><tr><td></td><td>1.0</td><td>0.47</td><td>0.24</td><td>21.81</td><td>0.276</td><td>0.270</td><td>5.35</td><td>0.52</td><td>2.77</td></tr><tr><td>+ FLOWCPO (β = 1, Ours)</td><td>3.0</td><td>0.70</td><td>0.56</td><td>22.46</td><td>0.297</td><td>0.295</td><td>5.41</td><td>1.02</td><td>3.23</td></tr><tr><td></td><td>4.5</td><td>0.70</td><td>0.59</td><td>22.36</td><td>0.294</td><td>0.294</td><td>5.38</td><td>1.02</td><td>3.22</td></tr></table>

Baselines. We compare against the two published approaches that are most directly comparable under the same fixed-data protocol:

• RFT (Rejection Sampling Fine-Tuning) [49, 5]: A positive-only offline baseline that can be viewed as a forward-KL special case, learning exclusively from preferred data and ignoring dispreferred data.

• FlowDPO [33]: A flow-based DPO baseline that uses preferred/dispreferred pairs through a reference-relative likelihood-ratio surrogate.

Evaluation Metrics. We evaluate two kinds of behavior. For targeted capabilities, we use taskspecific benchmarks: GenEval [12] for multi-concept compositional generation and an OCR-based metric for visual text rendering. For general preference alignment, we run inference on the DrawBench prompt set and report PickScore [23], CLIP Score [17], HPS v2.1 [48], Aesthetics [39], ImageReward [50], and UnifiedReward [47]. Because the in-domain general-preference subset is filtered with PickScore, CLIP Score, and HPS v2.1, we treat these as optimization-aligned metrics and use Aesthetics, ImageReward, and UnifiedReward as a cleaner check of transfer beyond the filtering pipeline.

## 5.2 RQ1: Main Results with In-Domain and Out-of-Domain Data

To evaluate method-level performance, we study three capability groups: semantic alignment, typographic generation, and general preference.

![](images/414489538ccb79241c1866816200ba22aa05d66f2a33be82d39bd088b5bbb0c9.jpg)  
Figure 2: Representative qualitative comparison of FLOWCPO and the baselines across three preference optimization tasks.

We summarize the main quantitative results in Tab. 3. The two regimes answer different questions, so we read them separately: the in-domain block evaluates in-domain offline fine-tuning, whereas the out-of-domain (OOD) block evaluates a single model trained once on an OOD offline dataset.

• In the in-domain setting, FLOWCPO has higher mean GenEval and OCR scores than RFT and FlowDPO. Tab. 3 aggregates separate task-specific fine-tunes for the three capability groups; FLOWCPO attains the best GenEval mean (0.84 at CFG 3.0) and the best OCR mean (0.87 at CFG 3.0), exceeding FlowDPO by 0.03 and 0.12, respectively. On general preference, the picture is narrower but still favorable: FLOWCPO is best or tied-best on PickScore, HPS v2.1, ImageReward, and UnifiedReward, while CLIP Score is effectively tied with FlowDPO. Since PickScore, CLIP Score, and HPS v2.1 also appear in the in-domain filtering pipeline, we put more weight on Aesthetics, ImageReward, and UnifiedReward, together with the qualitative comparison in Fig. 2 and the extended examples in App. D.2, when judging transfer beyond the optimization-aligned signals.

• In the out-of-domain setting, the advantage is metric-dependent. FLOWCPO gives the best GenEval score and remains competitive on OCR and UnifiedReward, whereas RFT performs better on several reward-model metrics. Thus, retaining both sides of each preference pair does not guarantee improvement when the fixed data are generated by models other than the reference model. This result motivates the regularization analysis in RQ2 and identifies out-of-domain data as an empirical boundary of the method.

## 5.3 RQ2: Effectiveness of Different Negative Regularization

We compare positive-only training with repulsive $( \lambda < 0 )$ and attractive $( \lambda > 0 )$ negative regularization under the unified framework.

Tab. 4 favors moderate attractive matching: $\beta = 0 . 5 , \lambda = 1$ achieves the best GenEval score (0.84), while $\beta = 1 , \lambda = 0 . 1$ gives the best OCR, ImageReward, and UnifiedReward. Repulsion with $\lambda = - 1$ sharply degrades all metrics, and both signs diverge at $| \lambda | = 1 0$ . Since training optimizes only GenEval, the other gains measure cross-metric transfer.

Table 4: Analysis of different regularization mechanisms on SD3.5-M under the unified divergence-based framework of equation 14, with all models trained only on GenEval. We consider three representative paradigms: positive-only regularization, which uses only preferred samples; repulsive negative regularization, which explicitly pushes the model away from dispreferred regions; and attractive negative regularization, which incorporates negative samples through distributional matching. For attractive negative regularization, we report two instantiations with different mixing coefficients β. Results are reported with and without Classifier-Free Guidance (CFG). Metrics are grouped into Concept, Typography, and General Preference for consistency with the main results table and to assess cross-metric generalization beyond the training signal. Best results are highlighted in bold and second-best results are underlined.
<table><tr><td rowspan="2">λ</td><td rowspan="2">CFG</td><td>Concept</td><td>Typography</td><td colspan="6">General Preference</td></tr><tr><td>GenEval ↑</td><td>OCR↑</td><td>PickScore ↑</td><td>CLIPSc. ↑</td><td>HPSv2.1 ↑</td><td>Aes. ↑</td><td>ImgRwd ↑</td><td>UniRwd ↑</td></tr><tr><td colspan="9">Positive-only regularization (RFT-like): µθ = vθ, νθ = 0</td></tr><tr><td>0</td><td>1.0</td><td>0.59</td><td>0.15</td><td>21.67</td><td>0.27</td><td>0.268</td><td>5.32</td><td>0.43</td><td>2.63</td></tr><tr><td>0</td><td>3.0</td><td>0.72</td><td>0.51</td><td>22.5</td><td>0.295</td><td>0.301</td><td>5.41</td><td>1.03</td><td>3.13</td></tr><tr><td>0</td><td>4.5</td><td>0.75</td><td>0.56</td><td>22.5</td><td>0.296</td><td>0.304</td><td>5.41</td><td>1.08</td><td>3.16</td></tr><tr><td colspan="9">Repulsive negative regularization (DPO-like): µθ = νθ = vθ</td><td></td></tr><tr><td>-0.1</td><td>1.0</td><td>0.64</td><td>0.15</td><td>21.76</td><td>0.277</td><td>0.270</td><td>5.30</td><td>0.53</td><td>2.66</td></tr><tr><td>-0.1</td><td>3.0</td><td>0.77</td><td>0.50</td><td>22.48</td><td>0.297</td><td>0.300</td><td>5.37</td><td>1.07</td><td>3.17</td></tr><tr><td>-0.1</td><td>4.5</td><td>0.76</td><td>0.58</td><td>22.47</td><td>0.297</td><td>0.304</td><td>5.40</td><td>1.10</td><td>3.17</td></tr><tr><td>-1</td><td>1.0</td><td>0.25</td><td>0.08</td><td>19.46</td><td>0.201</td><td>0.154</td><td>4.22</td><td>-1.66</td><td>1.40</td></tr><tr><td>-1</td><td>3.0</td><td>0.70</td><td>0.48</td><td>21.39</td><td>0.270</td><td>0.229</td><td>4.70</td><td>-0.28</td><td>2.51</td></tr><tr><td>-1</td><td>4.5</td><td>0.71</td><td>0.55</td><td>21.50</td><td>0.276</td><td>0.240</td><td>4.86</td><td>-0.01</td><td>2.65</td></tr><tr><td>-10</td><td></td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td></tr><tr><td colspan="9">Attractive negative regularization (CPO-like): µθ = vθ, νθ 2vold − vθ</td></tr><tr><td>0.1</td><td>1.0</td><td>0.64</td><td>0.17</td><td>21.72</td><td>0.278</td><td>0.266</td><td>5.24</td><td>0.50</td><td>2.71</td></tr><tr><td>0.1</td><td>3.0</td><td>0.77</td><td>0.54</td><td>22.47</td><td>0.297</td><td>0.301</td><td>5.35</td><td>1.06</td><td>3.19</td></tr><tr><td>0.1</td><td>4.5</td><td>0.77</td><td>0.59</td><td>22.48</td><td>0.298</td><td>0.304</td><td>5.38</td><td>1.14</td><td>3.23</td></tr><tr><td>1</td><td>1.0</td><td>0.78</td><td>0.25</td><td>21.74</td><td>0.280</td><td>0.258</td><td>5.23</td><td>0.57</td><td>2.74</td></tr><tr><td>1</td><td>3.0</td><td>0.78</td><td>0.51</td><td>21.69</td><td>0.291</td><td>0.266</td><td>5.09</td><td>0.76</td><td>2.96</td></tr><tr><td>1</td><td>4.5</td><td>0.67</td><td>0.55</td><td>21.20</td><td>0.285</td><td>0.245</td><td>4.89</td><td>0.43</td><td>2.75</td></tr><tr><td>10</td><td></td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td></tr><tr><td colspan="10">Attractive negative regularization (CPO-like): µθ = 0.5vold + 0.5vθ, νθ = 1.5vold − 0.5vθ</td></tr><tr><td>0.1 0.1</td><td>1.0</td><td>0.63</td><td>0.17</td><td>21.48 22.41</td><td>0.275 0.298</td><td>0.259</td><td>5.22 5.33</td><td>0.33 1.04</td><td>2.55</td></tr><tr><td>0.1</td><td>3.0</td><td>0.78</td><td>0.51</td><td></td><td></td><td>0.297</td><td></td><td>1.10</td><td>3.14</td></tr><tr><td></td><td>4.5</td><td>0.77</td><td>0.58</td><td>22.43</td><td>0.299</td><td>0.299</td><td>5.35</td><td></td><td>3.19</td></tr><tr><td>1</td><td>1.0</td><td>0.76</td><td>0.26</td><td>21.84</td><td>0.280</td><td>0.268</td><td>5.26</td><td>0.64</td><td>2.79</td></tr><tr><td>1</td><td>3.0</td><td>0.84</td><td>0.53</td><td>22.20</td><td>0.296</td><td>0.288</td><td>5.27</td><td>1.01</td><td>3.10</td></tr><tr><td>1</td><td>4.5</td><td>0.83</td><td>0.55</td><td>21.94</td><td>0.294</td><td>0.282</td><td>5.16</td><td>0.95</td><td>3.04</td></tr><tr><td>10</td><td>一</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td><td>Diverged</td></tr></table>

## 5.4 RQ3: Sensitivity of FLOWCPO to Its Core Hyperparameters

Fig. 9 favors moderate interpolation $( \beta \in [ 0 . 5 , 1 . 0 ] )$ , balanced negative weight (λ = 1), and slow EMA $( \eta = 0 . 9 9 )$ . Extreme weights destabilize training, while small weights weaken the negative contribution. We therefore use $\beta = 0 . 5 , \lambda = 1$ , and $\eta = 0 . 9 9$ as the default settings. Complete sweeps appear in App. D.3.2.

## 6 Conclusion and Limitations

We presented a divergence-based framework that separates sampling regime from divergence direction and derived FLOWCPO as its offline forward-KL instance. Coupled preferred and dispreferred branches enable regression on fixed preference pairs without online rollouts. Our loss decomposition also connects FLOWCPO to simplified FlowDPO and shows how the difference between the current model and its EMA enters the objective. The nonnegative loss avoids simplified FlowDPO’s potentially unbounded negative regression term.

Experiments show higher mean in-domain GenEval and OCR scores than RFT and FlowDPO, with mixed out-of-domain gains. The ablations favor moderate attractive matching of dispreferred samples, while excessively large weights can destabilize training.

The forward-KL bound assumes linear interpolation and uniform regularity. Loss nonnegativity alone does not guarantee stable training. We evaluate fine-tuning on SD3.5-M, so performance on other backbones remains to be established. Extending the bound to other paths and improving robustness across different data sources remain open challenges.

## References

[1] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.

[2] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International conference on artificial intelligence and statistics, pages 4447–4455. PMLR, 2024.

[3] Andreas Bergmeister, Stefanie Jegelka, Nikolas Nüsken, Carles Domingo-Enrich, and Jakiw Pidstrigach. Reinforce adjoint matching: Scaling rl post-training of diffusion and flow-matching models. arXiv preprint arXiv:2605.10759, 2026.

[4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024.

[5] Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu, and Haoxiang Wang. Bridging supervised learning and reinforcement learning in math reasoning. arXiv preprint arXiv:2505.18116v1, 2025.

[6] Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. Directly fine-tuning diffusion models on differentiable rewards. In The Twelfth International Conference on Learning Representations, 2024.

[7] Data Is Better Together. Open image preferences v1 results. https://huggingface.co/ datasets/data-is-better-together/open-image-preferences-v1-results, 2024. Hugging Face dataset, accessed March 26, 2026.

[8] Zheng Ding and Weirui Ye. Treegrpo: Tree-advantage grpo for online rl post-training of diffusion models. arXiv preprint arXiv:2512.08153, 2025.

[9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[10] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. In Advances in Neural Information Processing Systems, volume 36, pages 79858–79885, 2023.

[11] Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023. URL https: //openreview.net/forum?id=g97OHbQyk1.

[12] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

[13] Dongyoung Go, Tomasz Korbak, Germán Kruszewski, Jos Rozen, Nahyeon Ryu, and Marc Dymetman. Aligning language models with preferences through f-divergence minimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 11546–11583. PMLR, 2023. URL https: //proceedings.mlr.press/v202/go23a.html.

[14] Jiaqi Han, Mingjian Jiang, Yuxuan Song, Stefano Ermon, and Minkai Xu. f-po: Generalizing preference optimization with f-divergence minimization. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 1144–1152. PMLR, 2025. URL https: //proceedings.mlr.press/v258/han25a.html.

[15] Yansen Han, Hongxin Sun, and Tao Lin. When can conditional flow matching replace pointwise negative log-likelihood? arXiv preprint arXiv:2608.28010, 2026.

[16] Xiaoxuan He, Siming Fu, Yuke Zhao, Wanli Li, Jian Yang, Dacheng Yin, Fengyun Rao, and Bo Zhang. Tempflow-grpo: When timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324, 2025.

[17] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings ofthe 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021.

[18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[20] Peter Holderrieth and Ezra Erives. An introduction to flow matching and diffusion models. arXiv preprint arXiv:2506.02070, 2025.

[21] Jiwoo Hong, Sayak Paul, Noah Lee, Kashif Rasul, James Thorne, and Jongheon Jeong. Margin aware preference optimization for aligning diffusion models without reference. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 4744–4752, 2026. doi: 10.1609/aaai.v40i6.42476.

[22] Haozhe Ji, Cheng Lu, Yilin Niu, Pei Ke, Hongning Wang, Jun Zhu, Jie Tang, and Minlie Huang. Towards efficient exact optimization of language model alignment. arXiv preprint arXiv:2402.00856, 2024.

[23] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

[24] Chieh-Hsin Lai, Yang Song, Dongjun Kim, Yuki Mitsufuji, and Stefano Ermon. The principles of diffusion models. arXiv preprint arXiv:2510.21890, 2025.

[25] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018.

[26] Binxu Li, Minkai Xu, Jiaqi Han, Meihua Dang, and Stefano Ermon. Divergence minimization preference optimization for diffusion model alignment. arXiv preprint arXiv:2507.07510, 2025.

[27] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802v1, 2025.

[28] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025.

[29] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Mingxi Cheng, Ji Li, and Liang Zheng. Aesthetic post-training diffusion models from generic preferences with stepby-step preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13199–13208, 2025.

[30] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[31] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky TQ Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv preprint arXiv:2412.06264, 2024.

[32] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

[33] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.

[34] Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Maximum likelihood training for score-based diffusion ODEs by high-order denoising score matching. arXiv preprint arXiv:2206.08265, 2022.

[35] Calvin Luo. Understanding diffusion models: A unified perspective. arXiv preprint arXiv:2208.11970, 2022.

[36] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[37] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning textto-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739v2, 2023. Withdrawn in 2024; subsumed by arXiv:2407.08737.

[38] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

[39] Christoph Schuhmann. LAION-aesthetics. LAION blog, 2022. URL https://laion.ai/ blog/laion-aesthetics/.

[40] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

[41] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. Advances in neural information processing systems, 34: 1415–1428, 2021.

[42] Huashan Sun, Shengyi Liao, Yansen Han, Yu Bai, Yang Gao, Cheng Fu, Weizhou Shen, Fanqi Wan, Ming Yan, Ji Zhang, et al. Solopo: Unlocking long-context capabilities in llms via short-to-long preference optimization. arXiv preprint arXiv:2505.11166, 2025.

[43] Peng Sun, Yi Jiang, and Tao Lin. Unified continuous generative models. arXiv preprint arXiv:2505.07447, 2025.

[44] Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Rémi Munos, Mark Rowland, Pierre Harvey Richemond, Michal Valko, Bernardo Ávila Pires, and Bilal Piot. Generalized preference optimization: A unified approach to offline alignment. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 47725–47742. PMLR, 2024. URL https://proceedings.mlr. press/v235/tang24b.html.

[45] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

[46] Chaoqi Wang, Yibo Jiang, Chenghao Yang, Han Liu, and Yuxin Chen. Beyond reverse KL: Generalizing direct preference optimization with diverse divergence constraints. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=2cRzmWXK9N.

[47] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.

[48] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

[49] Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

[50] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

[51] Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025.

[52] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8941–8951, 2024.

[53] Shentao Yang, Tianqi Chen, and Mingyuan Zhou. A dense reward view on aligning textto-image diffusion with preference. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 55998–56032, 2024.

[54] Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. Slichf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

[55] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

1 Introduction 1   
2 Related Work 2   
3 Preliminaries   
3.1 Diffusion Models and Flow Matching   
3.2 RLHF and Direct Preference Optimization (DPO)   
3.3 Preference Optimization in Continuous-Time Models   
3.4 Forward and Reverse KL Divergences   
4 Methodology   
4.1 Probabilistic Reward Formulation   
4.2 Re-formalizing Reinforcement Learning from Human Feedback   
4.3 Unified View and Connections to Prior Work 6   
5 Experiments 6   
5.1 Experimental Setup 6   
5.2 RQ1: Main Results with In-Domain and Out-of-Domain Data 7   
5.3 RQ2: Effectiveness of Different Negative Regularization 8   
5.4 RQ3: Sensitivity of FLOWCPO to Its Core Hyperparameters 9   
6 Conclusion and Limitations 9   
A Theoretical Relation between Log-Likelihood and Flow Matching Loss 15   
A.1 Pointwise Relation between Log-Likelihood and Flow Matching Loss 15   
A.2 Expectation-level Relation between Log-Likelihood and Flow Matching Loss 15   
B Theoretical Proofs and Results 20   
B.1 Idealized Posteriors and Empirical Source Distributions 20   
B.2 Proof of Theorem 4.1 20   
B.3 An Idealized Interpretation of Interpolated Vector Fields 20   
B.4 Proof of Theorem 4.2 21   
B.5 Relation between FlowCPO and Simplified FlowDPO . 21   
C Connections to Existing Methods Under the Unified Divergence Framework 24   
C.1 Connection to FlowGRPO 24   
C.2 Connection to DiffusionNFT 24   
C.3 Connection to AWM 24   
C.4 Connection to RAM . 25   
C.5 Connection to SFT 25   
C.6 Connection to RFT 26   
C.7 Connection to FlowDPO 26   
D Experimental Details 28   
D.1 Experimental Details of Sec. 5.2 28   
D.2 Experimental Results on Optimization Targets . 29   
D.3 Additional Analyses Beyond Main Results . 30   
D.3.1 Understanding the Role of Negative Regularization 30   
D.3.2 Hyperparameter Sensitivity and Ablation 31   
D.4 Per-Prompt Diversity Evaluation 33   
E Broader Impacts and Release Considerations 34

## A Theoretical Relation between Log-Likelihood and Flow Matching Loss

In this section, we discuss the relationship between log-likelihood and flow matching loss. The theoretical results mainly follow from [41, 15, 34].

## A.1 Pointwise Relation between Log-Likelihood and Flow Matching Loss

The theoretical results in this subsection are from [15]. Fix a sample $x _ { 0 } \in \mathbb { R } ^ { d }$

$$
\mathbf { x } _ { t } = ( 1 - t ) x _ { 0 } + t \mathbf { x } _ { 1 } \sim q _ { t } ^ { x _ { 0 } } = { \mathcal { N } } ( ( 1 - t ) x _ { 0 } , t ^ { 2 } I _ { d } ) , \qquad \mathbf { x } _ { 1 } \sim { \mathcal { N } } ( 0 , I _ { d } ) .
$$

Let $\begin{array} { r } { u _ { t } ^ { x _ { 0 } } ( \mathbf { x } ) = \frac { \mathbf { x } - x _ { 0 } } { t } } \end{array}$ be the velocity field corresponding to $q _ { t } ^ { x _ { 0 } }$ . We compare this conditional path $( q _ { t } ^ { x _ { 0 } } , u _ { t } ^ { x _ { 0 } } )$ to the model path $( p _ { t } ^ { \theta } , v _ { \theta } )$ through the velocity and score gaps:

$$
\Delta v _ { t } ^ { \theta } ( \mathbf { x } ; x _ { 0 } ) : = v _ { \theta } ( \mathbf { x } , t ) - u _ { t } ^ { x _ { 0 } } ( \mathbf { x } ) , \qquad \Delta s _ { t } ^ { \theta } ( \mathbf { x } ; x _ { 0 } ) : = \nabla \log q _ { t } ^ { x _ { 0 } } ( \mathbf { x } ) - \nabla \log p _ { t } ^ { \theta } ( \mathbf { x } )
$$

we also define the following terms:

$$
\ell _ { \varepsilon } ( \theta ; x _ { 0 } ) : = \mathbb { E } _ { \mathbf { x } _ { \varepsilon } \sim q _ { \varepsilon } ^ { x _ { 0 } } } \bigl [ - \log p _ { \varepsilon } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) \bigr ] , \quad \mathcal { I } _ { w } ^ { [ \varepsilon , 1 ] } ( \theta ; x _ { 0 } ) : = \int _ { \varepsilon } ^ { 1 } w ( t ) \mathbb { E } _ { \mathbf { x } _ { t } \sim q _ { t } ^ { x _ { 0 } } } \bigl [ \| \Delta v _ { t } ^ { \theta } ( \mathbf { x } _ { t } ; x _ { 0 } ) \| ^ { 2 } \bigr ] d t
$$

Assumption A.1 (Pathwise regularity). For the fixed $x _ { 0 } ,$ , the densities $q _ { t } ^ { x _ { 0 } }$ and $p _ { t } ^ { \theta }$ are strictly positive and $C ^ { 1 }$ in x for $t \in \mathsf { \Gamma } ( 0 , 1 )$ Their continuity equations hold in strong form, t 7→ $\mathrm { K L } ( q _ { t } ^ { x _ { 0 } } \| p _ { t } ^ { \theta } )$ is differentiable, and the integrations by parts used below have no boundary terms.

Assumption A.2 (Endpoint regularity). The map $f _ { \theta } ( x , t ) : = - \log p _ { t } ^ { \theta } ( x )$ is jointly continuous at $( x _ { 0 } , 0 )$ . Moreover,for some $\varepsilon _ { 0 } > 0 , C > 0 ,$ , and m $\geq 1$

$$
| f _ { \theta } ( x , t ) | \leq C ( 1 + \| x \| ^ { m } ) , \qquad x \in \mathbb { R } ^ { d } , \quad t \in [ 0 , \varepsilon _ { 0 } ] .
$$

Theorem A.3 (Pointwise NLL is a CFM term plus residuals [15]). Suppose Assump. A.1 and Assump. A.2 hold, and suppose that $p _ { 1 } ^ { \theta } = q _ { 1 } ^ { x _ { 0 } } \overset { \cdot } { = } \mathcal { N } ( 0 , I _ { d } )$ . Then, for every positive weight w and every $\varepsilon \in ( 0 , 1 )$

$$
\ell _ { \varepsilon } ( \theta ; x _ { 0 } ) = H ( q _ { \varepsilon } ^ { x _ { 0 } } ) + \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { q _ { t } ^ { x _ { 0 } } } \big [ \langle \Delta v _ { t } ^ { \theta } ( \mathbf { x } _ { t } ; x _ { 0 } ) , \Delta s _ { t } ^ { \theta } ( \mathbf { x } _ { t } ; x _ { 0 } ) \rangle \big ] d t ,\tag{15}
$$

$$
- \log p _ { 0 } ^ { \theta } ( x _ { 0 } ) = H ( q _ { \varepsilon } ^ { x _ { 0 } } ) + \mathcal { J } _ { w } ^ { [ \varepsilon , 1 ] } ( \theta ; x _ { 0 } ) + \mathcal { G } _ { \varepsilon , w } ( \theta ; x _ { 0 } ) + \mathcal { B } _ { \varepsilon } ( \theta ; x _ { 0 } ) ,\tag{16}
$$

where

$$
\mathcal { G } _ { \varepsilon , w } ( \theta ; x _ { 0 } ) : = \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { q _ { t } ^ { x _ { 0 } } } \big [ \langle \Delta v _ { t } ^ { \theta } ( \mathbf { x } _ { t } ; x _ { 0 } ) , \Delta s _ { t } ^ { \theta } ( \mathbf { x } _ { t } ; x _ { 0 } ) - w ( t ) \Delta v _ { t } ^ { \theta } ( \mathbf { x } _ { t } ; x _ { 0 } ) \rangle \big ] d t ,
$$

$$
H ( q _ { \varepsilon } ^ { x _ { 0 } } ) = \frac { d } { 2 } \log ( 2 \pi e \varepsilon ^ { 2 } ) , \quad \mathcal { B } _ { \varepsilon } ( \theta ; x _ { 0 } ) : = - \log p _ { 0 } ^ { \theta } ( x _ { 0 } ) - \ell _ { \varepsilon } ( \theta ; x _ { 0 } ) , \quad \operatorname* { l i m } _ { \varepsilon \downarrow 0 } \mathcal { B } _ { \varepsilon } ( \theta ; x _ { 0 } ) = 0 .
$$

## A.2 Expectation-level Relation between Log-Likelihood and Flow Matching Loss

Throughout this subsection, let

$$
t \sim \mathrm { U n i f } ( 0 , 1 ) , \qquad \mathbf { x } _ { 0 } \sim p _ { \mathrm { d a t a } } , \qquad \mathbf { x } _ { 1 } \sim \mathcal { N } ( 0 , I _ { d } ) ,
$$

where $\mathbf { x } _ { \mathrm { 0 } }$ and $\mathbf { x } _ { 1 }$ are independent, and let

$$
\mathbf { x } _ { t } = ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } \sim q _ { t } .
$$

The marginal velocity field of $( q _ { t } ) _ { t \in [ 0 , 1 ] }$ is

$$
u _ { t } ( x ) : = \mathbb { E } [ { \mathbf { x } } _ { 1 } - { \mathbf { x } } _ { 0 } \mid { \mathbf { x } } _ { t } = x ] .
$$

Lemma A.4 (Optimal solution of Flow Matching Loss). Let us consider the following optimization problem:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \| v _ { \theta } ( ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \right]\tag{17}
$$

Then, for every $t \in ( 0 , 1 )$ , the optimal solution is given by

$$
v ^ { \star } ( x , t ) = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } = x \right]\tag{18}
$$

where ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$

Proof.

$$
\begin{array} { r l } & { \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \| v _ { \theta } ( ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] } \\ & { \quad = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - v ^ { \star } ( \mathbf { x } _ { t } , t ) + v ^ { \star } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] } \\ & { \quad = \mathbb { E } _ { t , \mathbf { x } _ { t } } [ \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - v ^ { \star } ( \mathbf { x } _ { t } , t ) + v ^ { \star } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \mid \mathbf { x } _ { t } \right] ] } \\ & { \qquad ( \mathrm { b y ~ t o w e r ~ p r o p e r t y ~ o f ~ e x p e c t a t i o n } ) } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { N o w , w e ~ p r o v e ~ t h a t ~ \mathbb { E } } _ { t , \mathbf { x } _ { t } } [ \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \left. v _ { \theta } ( \mathbf { x } _ { t } , t ) - v ^ { \star } ( \mathbf { x } _ { t } , t ) , v ^ { \star } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \right. | \mathbf { x } _ { t } ] \right] = 0 . } \\ & { \mathbb { E } _ { t , \mathbf { x } _ { t } } [ \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \left. v _ { \theta } ( \mathbf { x } _ { t } , t ) - v ^ { \star } ( \mathbf { x } _ { t } , t ) , v ^ { \star } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \right. | \mathbf { x } _ { t } \right] ] } \\ & { \quad = \mathbb { E } _ { t , \mathbf { x } _ { t } } [ \langle v _ { \theta } ( \mathbf { x } _ { t } , t ) - v ^ { \star } ( \mathbf { x } _ { t } , t ) , v ^ { \star } ( \mathbf { x } _ { t } , t ) - \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \left( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \right) | \mathbf { x } _ { t } \right] \rangle ] } \\ & { \qquad \mathrm { ( b y ~ t h e ~ l i n e a r i t y ~ o f ~ e x p e c t a t i o n ) } } \\ &  \quad = \mathbb { E } _ { t , \mathbf { x } _ { t } } [ \langle v _ { \theta } ( \mathbf { x } _ { t } , t ) - v ^ { \star } ( \mathbf { x } _ { t } , t ) , v ^ { \star } ( \mathbf { x } _ { t } , t ) - v ^ { \star } ( \mathbf { x } _ { t } , t ) \rangle ] \quad \mathrm { ( b y ~ } v ^ { \star } ( \mathbf { x } _ { t } , t ) = \mathbb { E } \end{array}
$$

Therefore,

$$
\begin{array} { r l } & { \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \| v _ { \theta } ( ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] } \\ & { \ = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - v ^ { \star } ( \mathbf { x } _ { t } , t ) \| _ { 2 } ^ { 2 } + \| v ^ { \star } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } ] } \end{array}
$$

Clearly, the optimal solution is attained when v<sub>θ</sub>(x, t) = v<sup>⋆</sup>(x, t).

Lemma A.5. Under the optimal solution $v ^ { \star } ( x , t )$ , we have:

$$
v ^ { \star } ( x , t ) = - \frac { 1 } { 1 - t } x - \frac { t } { 1 - t } \nabla \log q _ { t } ( x )\tag{19}
$$

where $\mathbf { x } _ { t } = ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } \sim q _ { t } ( x )$

Proof. Let $q _ { t } ^ { x _ { 0 } } ( x ) = q _ { t } ( x \mid x _ { 0 } )$ . Because ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ and $\mathbf { x } _ { 1 } \sim \mathcal { N } ( 0 , I )$ , then

$$
q _ { t } ^ { x _ { 0 } } ( \mathbf { x } _ { t } ) = \mathcal { N } ( ( 1 - t ) x _ { 0 } , t ^ { 2 } I )\tag{20}
$$

and we have:

$$
\nabla \log q _ { t } ^ { x _ { 0 } } ( { \mathbf x } _ { t } ) = - \frac { 1 } { t } { \mathbf x } _ { 1 }\tag{21}
$$

Therefore,

$$
\nabla \log q _ { t } ( \mathbf { x } _ { t } ) = \mathbb { E } [ \nabla \log q _ { t } ^ { x _ { 0 } } ( \mathbf { x } _ { t } ) \mid \mathbf { x } _ { t } ] = - { \frac { 1 } { t } } \mathbb { E } [ \mathbf { x } _ { 1 } \mid \mathbf { x } _ { t } ]\tag{22}
$$

Then, for the optimal solution $v ^ { \star } ( x , t )$ , we have:

$$
v ^ { \star } ( x , t ) = \mathbb { E } [ \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } = x ]\tag{23}
$$

$$
= \operatorname { \mathbb { E } } [ \mathbf { x } _ { 1 } - { \frac { \mathbf { x } _ { t } - t \mathbf { x } _ { 1 } } { 1 - t } } \mid \mathbf { x } _ { t } = x ]\tag{24}
$$

$$
= - { \frac { 1 } { 1 - t } } x + { \frac { 1 } { 1 - t } } \mathbb { E } [ \mathbf { x } _ { 1 } \mid \mathbf { x } _ { t } = x ]\tag{25}
$$

$$
= - { \frac { 1 } { 1 - t } } x - { \frac { t } { 1 - t } } \nabla \log q _ { t } ( x )\tag{26}
$$

Theorem A.6. Suppose Assump. A.1 and Assump. A.2 hold and let $p _ { \mathrm { d a t a } } ( \mathbf { x } _ { 0 } )$ be the data distribution. Then,

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { x } _ { 0 } \sim p _ { \mathrm { d a t a } } ( \mathbf { x } _ { 0 } ) } [ - \log p _ { 0 } ^ { \theta } ( \mathbf { x } _ { 0 } ) ] } \\ & { \leq H ( p _ { \mathrm { d a t a } } ) + \sqrt { \mathbb { E } \big [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \big ] \cdot \mathbb { E } \big [ \| \nabla \log q _ { t } ( \mathbf { x } _ { t } ) - \nabla \log p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) \| _ { 2 } ^ { 2 } \big ] } } \end{array}
$$

In the unrestricted, well-specified population setting, the marginal target field $v ^ { \star } ( \mathbf { x } _ { t } , t ) =$ $\mathbb { E } [ \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ]$ minimizes theflow matching loss and induces the data endpoint distribution, which also minimizes cross-entropy.

Proof. By Cauchy-Schwarz inequality, we have:

$$
( \mathbb { E } _ { t , { \mathbf { x } _ { 0 } } , { \mathbf { x } _ { t } } } [ \langle v _ { \theta } ( { \mathbf { x } _ { t } } , t ) - ( { \mathbf { x } _ { 1 } } - { \mathbf { x } _ { 0 } } ) , \nabla \log q _ { t } ( { \mathbf { x } _ { t } } ) - \nabla \log p _ { t } ^ { \theta } ( { \mathbf { x } _ { t } } ) \rangle ] ) ^ { 2 }\tag{27}
$$

$$
\begin{array} { r l } & { \leq \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { t } } [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } ] \cdot \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { t } } [ \| \nabla \log q _ { t } ( \mathbf { x } _ { t } ) - \nabla \log p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) \| _ { 2 } ^ { 2 } ] } \end{array}\tag{28}
$$

Then, by Thm. A.3, we have:

$$
\mathbb { E } _ { \mathbf { x } _ { 0 } \sim p _ { \mathrm { d a t a } } ( \mathbf { x } _ { 0 } ) } [ - \log p _ { 0 } ^ { \theta } ( \mathbf { x } _ { 0 } ) ]\tag{29}
$$

$$
= H ( p _ { \mathrm { d a t a } } ) + \mathbb { E } { \left[ \langle v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) , \nabla \log q _ { t } ( \mathbf { x } _ { t } ) - \nabla \log p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) \rangle \right] }\tag{30}
$$

$$
\leq H ( p _ { \mathrm { d a t a } } ) + ( \mathbb { E } \big [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \big ] \mathbb { E } \big [ \| \nabla \log q _ { t } ( \mathbf { x } _ { t } ) - \nabla \log p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) \| _ { 2 } ^ { 2 } \big ] ) ^ { 1 / 2 }\tag{31}
$$

By Lem. A.4, the unrestricted population minimizer is $v ^ { \star } ( \mathbf { x } _ { t } , t ) = \mathbb { E } [ \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \ | \ \mathbf { x } _ { t } ]$ . This field generates the prescribed marginal path $q _ { t } ,$ so $p _ { t } ^ { \theta ^ { \star } } = q _ { t }$ in the well-specified case and the scoregap term vanishes. Consequently the induced endpoint distribution equals $p _ { \mathrm { d a t a } } ,$ which minimizes cross-entropy. This population statement does not imply equality of the two objectives, nor identical minimizers in a restricted parameter class.

Theorem A.7 (Flow matching upper bound under uniform regularity). Suppose Assump. A.1 and Assump. A.2 hold, and let $p _ { 1 } ^ { \tilde { \theta } } = \mathbf { \hat { N } } ( 0 , I _ { d } )$ . Assume $M _ { 2 } : = \mathbf { \overline { { E } } } _ { p _ { \mathrm { d a t a } } } \mathbf { \bar { [ \left| x _ { 0 } \right| \left| \mathbf { \bar { 2 } } \right] } } <$ ∞ and that there exist constants $C _ { 0 } , C _ { 1 } , C _ { 2 } < \infty$ , independent ofθ, such that,for every $x \in \mathbb { R } ^ { d }$ and $t \in [ 0 , 1 ]$

$$
\begin{array} { r } { \| v _ { \theta } ( 0 , t ) \| \leq C _ { 0 } , \qquad \| \nabla v _ { \theta } ( x , t ) \| \leq C _ { 1 } , \qquad \| \nabla ^ { \top } \nabla \cdot v _ { \theta } ( x , t ) \| \leq C _ { 2 } . } \end{array}\tag{32}
$$

Assume additionally that the data-path score is uniformly square-integrable,

$$
C _ { q } : = \operatorname* { s u p } _ { t \in ( 0 , 1 ] } \mathbb { E } _ { q _ { t } } [ \| \nabla \log q _ { t } ( \mathbf { x } _ { t } ) \| ^ { 2 } ] < \infty .\tag{33}
$$

Then, there exists a constant $C _ { s } < \infty ,$ , independent of θ, such that:

$$
\mathbb { E } _ { \mathbf { x } _ { 0 } \sim p _ { \mathrm { d a t a } } } [ - \log p _ { 0 } ^ { \theta } ( \mathbf { x } _ { 0 } ) ] \le \mathbb { E } \big [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } \big ] + H ( p _ { \mathrm { d a t a } } ) + \frac { C _ { s } } { 4 } .\tag{34}
$$

In the unrestricted, well-specified population setting, the marginal target field minimizes both sides.

Proof. Let $s _ { t } ^ { \theta } ( x ) = \nabla \log p _ { t } ^ { \theta } ( x )$ , then we consider the continuity equation:

$$
\partial _ { t } p _ { t } ^ { \theta } ( x ) + \nabla \cdot ( p _ { t } ^ { \theta } ( x ) v _ { \theta } ( x , t ) ) = 0\tag{35}
$$

$$
\partial _ { t } p _ { t } ^ { \theta } ( x ) + \nabla p _ { t } ^ { \theta } ( x ) \cdot \boldsymbol { v } _ { \theta } ( x , t ) + p _ { t } ^ { \theta } ( x ) \nabla \cdot \boldsymbol { v } _ { \theta } ( x , t ) = 0\tag{36}
$$

$$
\begin{array} { r } { \partial _ { t } \log p _ { t } ^ { \theta } ( x ) + v _ { \theta } ( x , t ) \cdot \nabla \log p _ { t } ^ { \theta } ( x ) + \nabla \cdot v _ { \theta } ( x , t ) = 0 } \end{array}\tag{37}
$$

Taking $\nabla$ on both sides, we get:

$$
\begin{array} { r } { \partial _ { t } \nabla \log p _ { t } ^ { \theta } ( x ) + \nabla v _ { \theta } ( x , t ) \cdot \nabla \log p _ { t } ^ { \theta } ( x ) + v _ { \theta } ( x , t ) \nabla ^ { \top } \nabla \log p _ { t } ^ { \theta } ( x ) + \nabla ^ { \top } \nabla \cdot v _ { \theta } ( x , t ) = 0 } \end{array}\tag{38}
$$

$$
\begin{array} { r } { \partial _ { t } s _ { t } ^ { \theta } ( x ) + \nabla v _ { \theta } ( x , t ) \cdot s _ { t } ^ { \theta } ( x ) + v _ { \theta } ( x , t ) \nabla ^ { \top } s _ { t } ^ { \theta } ( x ) + \nabla ^ { \top } \nabla \cdot v _ { \theta } ( x , t ) = 0 } \end{array}\tag{39}
$$

Let $X _ { t }$ denote the trajectory, then we have:

$$
\frac { d } { d t } s _ { t } ^ { \theta } ( X _ { t } ) + \nabla v _ { \theta } ( X _ { t } , t ) \cdot s _ { t } ^ { \theta } ( X _ { t } ) + \nabla ^ { \top } \nabla \cdot v _ { \theta } ( X _ { t } , t ) = 0\tag{40}
$$

$$
\frac { d } { d t } s _ { t } ^ { \theta } ( X _ { t } ) = - \nabla v _ { \theta } ( X _ { t } , t ) \cdot s _ { t } ^ { \theta } ( X _ { t } ) - \nabla ^ { \top } \nabla \cdot v _ { \theta } ( X _ { t } , t )\tag{41}
$$

Therefore, we need to control $\| \nabla v _ { \theta } \|$ and $\| \nabla ^ { \top } \nabla \cdot v _ { \theta } \|$ . We assume $v _ { \theta }$ is smooth and constrained in a compact set, then we can have:

$$
\| \nabla v _ { \theta } \| \le C _ { 1 }\tag{42}
$$

$$
\| \nabla ^ { \top } \nabla \cdot v _ { \theta } \| \leq C _ { 2 }\tag{43}
$$

We further assume that $\| v _ { \theta } ( 0 , t ) \| \leq C _ { 0 }$ uniformly over the parameter set. Then, we define $g ( \lambda ) =$ $v _ { \theta } ( \lambda x , t ) , \lambda \in [ 0 , 1 ]$ , and we have:

$$
g ( 1 ) - g ( 0 ) = \int _ { 0 } ^ { 1 } \frac { d } { d \lambda } g ( \lambda ) d \lambda\tag{44}
$$

This is actually:

$$
v _ { \theta } ( x , t ) - v _ { \theta } ( 0 , t ) = \int _ { 0 } ^ { 1 } \nabla v _ { \theta } ( \lambda x , t ) x d \lambda\tag{45}
$$

Therefore, we have:

$$
\| v _ { \theta } ( x , t ) \| = \| v _ { \theta } ( 0 , t ) + \int _ { 0 } ^ { 1 } \nabla v _ { \theta } ( \lambda x , t ) x d \lambda \|\tag{46}
$$

$$
\leq \| v _ { \theta } ( 0 , t ) \| + \| \int _ { 0 } ^ { 1 } \nabla v _ { \theta } ( \lambda x , t ) x d \lambda \|\tag{47}
$$

$$
\leq C _ { 0 } + \int _ { 0 } ^ { 1 } \| \nabla v _ { \theta } ( \lambda x , t ) \| \| x \| d \lambda\tag{48}
$$

$$
\leq C _ { 0 } + C _ { 1 } \| x \|\tag{49}
$$

Let $X _ { r }$ be the characteristic starting from $X _ { t } = x .$ , namely,

$$
X _ { r } = x + \int _ { t } ^ { r } v _ { \theta } ( X _ { s } , s ) d s , \qquad r \in [ t , 1 ] .\tag{50}
$$

Using the linear-growth bound above, we obtain

$$
\| X _ { r } \| \leq \| x \| + \int _ { t } ^ { r } \| v _ { \theta } ( X _ { s } , s ) \| d s\tag{51}
$$

$$
\leq \| x \| + \int _ { t } ^ { r } ( C _ { 0 } + C _ { 1 } \| X _ { s } \| ) d s .\tag{52}
$$

Therefore, Grönwall’s inequality gives

$$
\left\| X _ { r } \right\| \leq e ^ { C _ { 1 } ( r - t ) } \| x \| + \frac { C _ { 0 } } { C _ { 1 } } \big ( e ^ { C _ { 1 } ( r - t ) } - 1 \big ) , \qquad r \in [ t , 1 ] ,\tag{53}
$$

Since $p _ { 1 } ^ { \theta } = \mathcal { N } ( 0 , I )$ , we have $s _ { 1 } ^ { \theta } ( x ) = - x$ . Then, by equation 41, we have:

$$
\frac { d } { d t } s _ { t } ^ { \theta } ( X _ { t } ) = - \nabla v _ { \theta } ( X _ { t } , t ) \cdot s _ { t } ^ { \theta } ( X _ { t } ) - \nabla ^ { \top } \nabla \cdot v _ { \theta } ( X _ { t } , t )\tag{54}
$$

$$
\int _ { t } ^ { 1 } d s _ { t } ^ { \theta } ( X _ { t } ) = \int _ { t } ^ { 1 } ( - \nabla v _ { \theta } ( X _ { t } , t ) \cdot s _ { t } ^ { \theta } ( X _ { t } ) - \nabla ^ { \top } \nabla \cdot v _ { \theta } ( X _ { t } , t ) ) d t\tag{55}
$$

$$
s _ { 1 } ^ { \theta } ( X _ { 1 } ) - s _ { t } ^ { \theta } ( X _ { t } ) = \int _ { t } ^ { 1 } ( - \nabla v _ { \theta } ( X _ { t } , t ) \cdot s _ { t } ^ { \theta } ( X _ { t } ) - \nabla ^ { \top } \nabla \cdot v _ { \theta } ( X _ { t } , t ) ) d t\tag{56}
$$

$$
\| s _ { t } ^ { \theta } ( X _ { t } ) \| \leq \| s _ { 1 } ^ { \theta } ( X _ { 1 } ) \| + \int _ { t } ^ { 1 } \left( C _ { 1 } \| s _ { r } ^ { \theta } ( X _ { r } ) \| + C _ { 2 } \right) d r\tag{57}
$$

$$
= \| X _ { 1 } \| + \int _ { t } ^ { 1 } \left( C _ { 1 } \| s _ { r } ^ { \theta } ( X _ { r } ) \| + C _ { 2 } \right) d r .\tag{58}
$$

Applying Grönwall’s inequality once more and using equation 53 gives

$$
\| s _ { t } ^ { \theta } ( x ) \| \leq e ^ { C _ { 1 } ( 1 - t ) } \| X _ { 1 } \| + \frac { C _ { 2 } } { C _ { 1 } } \big ( e ^ { C _ { 1 } ( 1 - t ) } - 1 \big )\tag{59}
$$

$$
\leq K _ { 0 } + K _ { 1 } \| x \| ,\tag{60}
$$

where $K _ { 0 } , K _ { 1 } < \infty$ depend only on $C _ { 0 } , C _ { 1 } , C _ { 2 }$ and are independent of θ. If $M _ { 2 } : = \mathbb { E } _ { p _ { \mathrm { d a t a } } } [ \| \mathbf { x } _ { 0 } \| ^ { 2 } ] <$ ∞, then

$$
\mathbb { E } [ \| \mathbf { x } _ { t } \| ^ { 2 } ] = \mathbb { E } [ \| ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } \| ^ { 2 } ]\tag{61}
$$

$$
\begin{array} { r l } { \mathbf { \tau } } & { { } = \mathbb { E } [ ( 1 - t ) ^ { 2 } \| \mathbf { x } _ { 0 } \| ^ { 2 } + t ^ { 2 } \| \mathbf { x } _ { 1 } \| ^ { 2 } + 2 t ( 1 - t ) \langle \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } \rangle ] } \end{array}\tag{62}
$$

$$
\begin{array} { r l } { \mathbf { \Psi } } & { { } = ( 1 - t ) ^ { 2 } \mathbb { E } [ \| \mathbf { x } _ { 0 } \| ^ { 2 } ] + t ^ { 2 } \mathbb { E } [ \| \mathbf { x } _ { 1 } \| ^ { 2 } ] + 2 t ( 1 - t ) \mathbb { E } [ \langle \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } \rangle ] } \end{array}\tag{63}
$$

$$
= ( 1 - t ) ^ { 2 } \mathbb { E } [ \| \mathbf { x _ { 0 } } \| ^ { 2 } ] + t ^ { 2 } \mathbb { E } [ \| \mathbf { x _ { 1 } } \| ^ { 2 } ] + 2 t ( 1 - t ) \langle \mathbb { E } [ \mathbf { x _ { 0 } } ] , \mathbb { E } [ \mathbf { x _ { 1 } } ] \rangle ] \quad ( \mathbf { b y ~ x _ { 0 } ~ \bot ~ x _ { 1 } } )\tag{64}
$$

$$
\begin{array} { r l } { = ( 1 - t ) ^ { 2 } \mathbb { E } \| { \bf x } _ { 0 } \| ^ { 2 } + t ^ { 2 } \mathbb { E } \| { \bf x } _ { 1 } \| ^ { 2 } } & { { } ( { \bf b } { \bf y } \mathbb { E } [ { \bf x } _ { 1 } ] = 0 ) } \end{array}\tag{65}
$$

$$
\begin{array} { r l } { \mathbf { \sigma } = ( 1 - t ) ^ { 2 } M _ { 2 } + t ^ { 2 } d } & { { } ( \mathbf { b y } \mathbf { x } _ { 1 } \sim \mathcal { N } ( 0 , I _ { d } ) ) } \end{array}\tag{66}
$$

$$
\leq M _ { 2 } + d \quad ( { \tt b y } t \in [ 0 , 1 ] )\tag{67}
$$

and consequently

$$
\operatorname* { s u p } _ { \theta , t \in [ 0 , 1 ] } \mathbb { E } _ { q _ { t } } [ \| \nabla \log p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) \| ^ { 2 } ] \leq 2 K _ { 0 } ^ { 2 } + 2 K _ { 1 } ^ { 2 } ( M _ { 2 } + d ) = : C _ { p } < \infty .\tag{68}
$$

It remains to control the score of the data path. By the identity in Lem. A.5,

$$
\nabla \log q _ { t } ( \mathbf { x } _ { t } ) = \mathbb { E } \left[ - \frac { \mathbf { x } _ { 1 } } { t } \Big | \mathbf { x } _ { t } \right] .\tag{69}
$$

Hence, on every truncated interval $t \in [ \varepsilon , 1 ]$

$$
\mathbb { E } _ { q _ { t } } [ \| \nabla \log q _ { t } ( \mathbf { x } _ { t } ) \| ^ { 2 } ] = \mathbb { E } _ { q _ { t } } [ \| - \frac { 1 } { t } \mathbb { E } [ \mathbf { x } _ { 1 } \mid \mathbf { x } _ { t } ] \| ^ { 2 } ]\tag{70}
$$

$$
= \frac { 1 } { t ^ { 2 } } \mathbb { E } _ { q _ { t } } [ \| \mathbb { E } [ \mathbf { x } _ { 1 } \mid \mathbf { x } _ { t } ] \| ^ { 2 } ]\tag{71}
$$

$$
\leq \frac { 1 } { t ^ { 2 } } \mathbb { E } _ { q _ { t } } [ \mathbb { E } [ \left\| \mathbf { x } _ { 1 } \right\| ^ { 2 } | \mathbf { x } _ { t } ] ]\tag{72}
$$

$$
= { \frac { 1 } { t ^ { 2 } } } \mathbb { E } [ \| { \mathbf { x } } _ { 1 } \| ^ { 2 } ] \quad ( { \mathrm { b y ~ t o w e r ~ p r o p e r t y } } )\tag{73}
$$

$$
\leq { \frac { d } { \varepsilon ^ { 2 } } } .\tag{74}
$$

For the full interval, we use the endpoint integrability condition in the theorem,

$$
C _ { q } : = \operatorname* { s u p } _ { t \in ( 0 , 1 ] } \mathbb { E } _ { q _ { t } } [ \| \nabla \log q _ { t } ( \mathbf { x } _ { t } ) \| ^ { 2 } ] < \infty .\tag{75}
$$

Combining the last two bounds yields

$$
\begin{array} { r } { \mathbb { E } \big [ \| \nabla \log q _ { t } ( \mathbf { x } _ { t } ) - \nabla \log p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) \| ^ { 2 } \big ] \leq 2 ( C _ { q } + C _ { p } ) = : C _ { s } < \infty , } \end{array}\tag{76}
$$

uniformly over θ. Substituting this estimate into Thm. $_ { \mathrm { A . 6 , } }$ and then using Young’s inequality, gives, for every $\kappa > 0$

$$
\mathbb { E } _ { \mathbf { x } _ { 0 } \sim p _ { \mathrm { d a t a } } } [ - \log p _ { 0 } ^ { \theta } ( \mathbf { x } _ { 0 } ) ] \leq H ( p _ { \mathrm { d a t a } } ) + \sqrt { C _ { s } \mathbb { E } [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } ] }\tag{77}
$$

$$
\leq \kappa \mathbb { E } [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } ] + H ( p _ { \mathrm { d a t a } } ) + \frac { C _ { s } } { 4 \kappa } .\tag{78}
$$

Taking $\kappa = 1$ proves the bound. The population-minimizer statement follows from Thm. A.6 under the stated well-specified interpretation. □

## B Theoretical Proofs and Results

## B.1 Idealized Posteriors and Empirical Source Distributions

The main text defines $\pi ^ { + }$ and $\pi ^ { - }$ as posterior target distributions induced by the binary optimality variable. In offline preference optimization, however, the data available for training are usually fixed empirical source distributions $\hat { \pi } ^ { + }$ and $\hat { \pi } ^ { - }$ constructed by a separate pipeline, such as best-of-N winner/loser selection from a frozen generator. The formal identities in the main text only require that the source distributions be fixed during optimization. The practical modeling question is therefore whether the offline construction is a faithful enough surrogate for the idealized targets. In our experiments, the data construction pipeline is frozen, so the resulting preferred and dispreferred distributions remain stationary throughout training.

## B.2 Proof of Theorem 4.1

Proof. For each context c, equation 9 gives

$$
r ( \mathbf { x } _ { 0 } , c ) = { \frac { 1 } { 2 \omega } } \left[ \log \pi ^ { + } ( \mathbf { x } _ { 0 } \mid c ) - \log \pi ^ { - } ( \mathbf { x } _ { 0 } \mid c ) + \log { \frac { p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) } { p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) } } \right] .
$$

Taking expectation under $\pi _ { \boldsymbol { \theta } } ( \cdot \mid c )$ yields

$$
\begin{array} { l } { \mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot | c ) } \big [ r ( \mathbf { x } _ { 0 } , c ) \big ] = \displaystyle \frac { 1 } { 2 \omega } \mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot | c ) } \big [ \log \pi ^ { + } ( \mathbf { x } _ { 0 } \mid c ) - \log \pi ^ { - } ( \mathbf { x } _ { 0 } \mid c ) \big ] } \\ { \displaystyle \qquad + \frac { 1 } { 2 \omega } \log \frac { p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) } { p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) } . } \end{array}
$$

The second term depends only on $c ,$ hence is irrelevant for optimization over θ. For the first term,

$$
\mathbb { E } _ { \pi _ { \theta } } [ \log \pi ^ { + } ] = - { \mathcal { D } } _ { K L } ( \pi _ { \theta } \| \pi ^ { + } ) - { \mathcal { H } } ( \pi _ { \theta } ) ,
$$

$$
\mathbb { E } _ { \pi _ { \theta } } [ \log \pi ^ { - } ] = - \mathcal { D } _ { K L } ( \pi _ { \theta } \| \pi ^ { - } ) - \mathcal { H } ( \pi _ { \theta } ) ,
$$

where $\mathcal { H } ( \pi _ { \theta } )$ is the conditional differential entropy of $\pi _ { \boldsymbol { \theta } } ( \cdot \mid c )$ . Subtracting the two identities cancel the entropy term:

$$
\mathbb { E } _ { \pi _ { \theta } } [ \log { \pi ^ { + } } - \log { \pi ^ { - } } ] = - { \mathcal { D } _ { K L } } ( \pi _ { \theta } \| { \pi ^ { + } } ) + { \mathcal { D } _ { K L } } ( \pi _ { \theta } \| { \pi ^ { - } } ) .
$$

Taking expectation over c and dropping the positive constant $\frac { 1 } { 2 \omega }$ proves the claim.

## B.3 An Idealized Interpretation of Interpolated Vector Fields

Proposition 1 of [20] gives the following relation when a velocity field and a score describe the same Gaussian marginal path (see also Lem. A.5):

$$
u _ { t } ( { \bf x } ) = - \frac { t } { 1 - t } \nabla \log p _ { t } ( { \bf x } ) - \frac { 1 } { 1 - t } { \bf x } .\tag{79}
$$

For an idealized interpretation, suppose that both $v _ { \theta }$ and $v _ { \mathrm { o l d } }$ satisfy this compatibility condition:

$$
v _ { \theta } ( \mathbf { x } _ { t } , t ) = - \frac { t } { 1 - t } \nabla \log p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) - \frac { 1 } { 1 - t } \mathbf { x } _ { t } ,\tag{80}
$$

$$
v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } , t ) = - \frac { t } { 1 - t } \nabla \log p _ { t } ^ { \mathrm { o l d } } ( \mathbf { x } _ { t } ) - \frac { 1 } { 1 - t } \mathbf { x } _ { t } .\tag{81}
$$

Then the positive interpolation satisfies

$$
v ^ { + } ( \mathbf { x } _ { t } , t ) = v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } , t ) + \beta ( v _ { \theta } ( \mathbf { x } _ { t } , t ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } , t ) )\tag{82}
$$

$$
= - \frac { t } { 1 - t } \nabla \log \left[ p _ { t } ^ { \mathrm { o l d } } ( \mathbf { x } _ { t } ) \left( \frac { p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) } { p _ { t } ^ { \mathrm { o l d } } ( \mathbf { x } _ { t } ) } \right) ^ { \beta } \right] - \frac { 1 } { 1 - t } \mathbf { x } _ { t } .\tag{83}
$$

Similarly,

$$
v ^ { - } ( \mathbf { x } _ { t } , t ) = v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } , t ) - \beta ( v _ { \theta } ( \mathbf { x } _ { t } , t ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } , t ) )\tag{84}
$$

$$
= - \frac { t } { 1 - t } \nabla \log \left[ p _ { t } ^ { \mathrm { o l d } } ( \mathbf { x } _ { t } ) \left( \frac { p _ { t } ^ { \mathrm { o l d } } ( \mathbf { x } _ { t } ) } { p _ { t } ^ { \theta } ( \mathbf { x } _ { t } ) } \right) ^ { \beta } \right] - \frac { 1 } { 1 - t } \mathbf { x } _ { t } .\tag{85}
$$

Thus, at each fixed $t , v ^ { + }$ and $v ^ { - }$ admit a score-field interpretation analogous to the CFG [35, 18] interpolation $\begin{array} { r } { p _ { t } ( \mathbf { x } _ { t } , \boldsymbol { \vartheta } ) \left( \frac { p _ { t } ( \mathbf { x } _ { t } , c ) } { p _ { t } ( \mathbf { x } _ { t } , \boldsymbol { \vartheta } ) } \right) ^ { \beta } } \end{array}$ over condition space.

## B.4 Proof of Theorem 4.2

Proof of Thm. 4.2. Expanding each KL term in equation 11 into cross-entropy plus entropy gives

$$
\mathcal { L } _ { \mathrm { O f f i n e } } ( \theta ) = \mathbb { E } _ { c } \Big [ \mathbb { E } _ { \mathbf { x } _ { 0 } ^ { + } \sim \pi ^ { + } ( \cdot | c ) } [ - \log \pi _ { \theta } ^ { + } ( \mathbf { x } _ { 0 } ^ { + } \mid c ) ] + \lambda \mathbb { E } _ { \mathbf { x } _ { 0 } ^ { - } \sim \pi ^ { - } ( \cdot | c ) } [ - \log \pi _ { \theta } ^ { - } ( \mathbf { x } _ { 0 } ^ { - } \mid c ) ] \Big ] + C _ { \mathrm { e n t } } ,
$$

where $C _ { \mathrm { e n t } } = - \mathbb { E } _ { c } [ \mathcal { H } ( \pi ^ { + } ( \cdot \mid c ) ) + \lambda \mathcal { H } ( \pi ^ { - } ( \cdot \mid c ) ) ]$ is independent of θ.

From Thm. A.7, we have:

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { x } _ { 0 } \sim p _ { \mathrm { d a t a } } } [ - \log p _ { 0 } ^ { \theta } ( \mathbf { x } _ { 0 } ) ] \le \mathbb { E } \big [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } \big ] + C . } \end{array}\tag{86}
$$

Therefore, we have:

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { x } _ { 0 } ^ { + } \sim \pi ^ { + } ( \cdot | c ) } [ - \log \pi _ { \theta } ^ { + } ( \mathbf { x } _ { 0 } ^ { + } \mid c ) ] \leq C _ { + } ( c ) + \mathbb { E } _ { \mathbf { x } _ { 0 } ^ { + } \sim \pi ^ { + } ( \cdot | c ) } [ \| \mu _ { \theta } ( \mathbf { x } _ { t } ^ { + } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { + } \mid \mathbf { x } _ { 0 } ^ { + } ) \| _ { 2 } ^ { 2 } ] , } \end{array}
$$

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { x } _ { 0 } ^ { - } \sim \pi ^ { - } ( \cdot | c ) } [ - \log \pi _ { \theta } ^ { - } ( \mathbf { x } _ { 0 } ^ { - } \mid c ) ] \leq C _ { - } ( c ) + \mathbb { E } _ { \mathbf { x } _ { 0 } ^ { - } \sim \pi ^ { - } ( \cdot | c ) } [ \| \nu _ { \theta } ( \mathbf { x } _ { t } ^ { - } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { - } \mid \mathbf { x } _ { 0 } ^ { - } ) \| _ { 2 } ^ { 2 } ] , } \end{array}
$$

Then,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { O f f i i n e } } ( \theta ) \leq \mathbb { E } _ { c , \mathbf { x } _ { 0 } ^ { + } \sim \pi ^ { + } ( \cdot \vert c ) } [ \| \mu _ { \theta } ( \mathbf { x } _ { t } ^ { + } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { + } \mid \mathbf { x } _ { 0 } ^ { + } ) \| _ { 2 } ^ { 2 } ] } \\ & { \qquad + \lambda \cdot \mathbb { E } _ { c , \mathbf { x } _ { 0 } ^ { - } \sim \pi ^ { - } ( \cdot \vert c ) } [ \| \nu _ { \theta } ( \mathbf { x } _ { t } ^ { - } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { - } \mid \mathbf { x } _ { 0 } ^ { - } ) \| _ { 2 } ^ { 2 } ] + C o n s t a n t . } \end{array}
$$

This establishes an upper-bound surrogate for equation 11. If the two branch fields can simultaneously realize their population targets, then both branch endpoint distributions match their corresponding sources. □

## B.5 Relation between FlowCPO and Simplified FlowDPO

We isolate the contrastive core of FlowDPO by omitting its sigmoid weighting and frozen-reference terms. Using the preferred/dispreferred notation from equation 13, the simplified objective is

$$
\displaystyle \operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { F l o w D P O } } ^ { \mathrm { s i m p l e } } ( \theta ) = \mathbb { E } \Big [ \| v _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - ( \epsilon - \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } - \| v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - ( \epsilon - \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } \Big ] .\tag{87}
$$

It captures the per-pair contrastive gradient direction, without retaining the full FlowDPO objective. Below, E averages over the same fixed offline pairs and sampled $t ,$ ϵ as in equation 13. We set $\lambda = 1$ to match the unweighted preference comparison above.

Theorem B.1 (Relation between FlowCPO and Simplified FlowDPO). Let $\beta > 0 , \lambda = 1$ , and hold $v _ { \mathrm { o l d } } ~ f i x e d$ during each gradient step. Assuming the displayed expectations are finite, the FlowCPO loss satisfies

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { F L o w C P O } } ( \theta ) = \beta \mathcal { L } _ { \mathrm { F l o w D P O } } ^ { \mathrm { s i m p l e } } ( \theta ) } \\ & { \qquad + \beta ( \beta - 1 ) \mathbb { E } \big [ \| v _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) \| _ { 2 } ^ { 2 } \big ] } \\ & { \qquad + \beta ( \beta + 1 ) \mathbb { E } \big [ \| v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) \| _ { 2 } ^ { 2 } \big ] + C , } \end{array}\tag{88}
$$

where $C$ is independent of θ for the current detached $v _ { \mathrm { o l d } }$

Proof. The preferred branch uses $\mathbf { x } _ { t } ^ { w }$ and target $u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) = \epsilon - \mathbf { x } _ { 0 } ^ { w }$ , while the dispreferred branch uses $\mathbf { x } _ { t } ^ { l }$ and target $u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) = \epsilon - \mathbf { x } _ { 0 } ^ { l }$ . We expand each branch separately.

Preferred branch. Substituting $\mu _ { \theta } = ( 1 - \beta ) v _ { \mathrm { o l d } } + \beta v _ { \theta }$ and collecting terms gives

$$
\begin{array} { r } { = { v _ { \mathrm { o l d } } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - { u _ { t } } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) + \beta \left[ v _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) \right] . } \end{array}
$$

Expanding the squared norm yields

$$
\begin{array} { r l } & { \| \mu _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } } \\ & { \quad = \| v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad + \beta ^ { 2 } \| v _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) \| _ { 2 } ^ { 2 } } \\ & { \qquad + 2 \beta \Big \langle v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) , v _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) \Big \rangle . } \end{array}
$$

The residual of the current model is the sum of the EMA residual and $v _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c )$ Expanding the squared norm of this sum and solving for the cross term gives

$$
\begin{array} { r l } & { 2 \Big \langle v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) , v _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) \Big \rangle } \\ & { \quad = \| v _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad - \| v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } - \| v _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

Substituting this identity into the expansion gives coefficients $\beta , \beta ^ { 2 } - \beta = \beta ( \beta - 1 )$ , and $1 - \beta$ for the three squared errors, respectively:

$$
\begin{array} { r l } & { \| \mu _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } = \beta \| v _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } } \\ & { \phantom { \| } + \beta ( \beta - 1 ) \| v _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) \| _ { 2 } ^ { 2 } } \\ & { \phantom { \| } + ( 1 - \beta ) \| v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { w } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } \mid \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

Dispreferred branch. Substituting $\nu _ { \theta } = ( 1 + \beta ) v _ { \mathrm { o l d } } - \beta v _ { \theta }$ now gives

$$
\begin{array} { r l } & { \nu _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) } \\ & { \quad = v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) - \beta \big [ v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) \big ] . } \end{array}
$$

The minus sign changes the sign of the cross term, while the coefficient of the squared distance remains $\beta ^ { 2 }$ :

$$
\begin{array} { r l } & { \| \nu _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } } \\ & { \quad = \| v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } } \\ & { \quad \quad + \beta ^ { 2 } \| v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) \| _ { 2 } ^ { 2 } } \\ & { \quad \quad - 2 \beta \Big \langle v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) , v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) \Big \rangle . } \end{array}
$$

Evaluating the same norm identity on the dispreferred input gives

$$
\begin{array} { r l } & { 2 \Big \langle v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) , v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) \Big \rangle } \\ & { \quad = \| v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } } \\ & { \quad \quad - \| v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } - \| v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

Substituting this identity gives coefficients $- \beta , \beta ^ { 2 } + \beta = \beta ( \beta + 1 )$ , and $1 + \beta \mathrm { : }$

$$
\begin{array} { r l } & { \| \nu _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } = - \beta \| v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad + \beta ( \beta + 1 ) \| v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) \| _ { 2 } ^ { 2 } } \\ & { \qquad + ( 1 + \beta ) \| v _ { \mathrm { o l d } } ( \mathbf { x } _ { t } ^ { l } , t , c ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } \mid \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

Take expectations over the preferred and dispreferred samples, respectively, and add. The first terms give $\beta { \mathcal { L } } _ { \mathrm { F l o w D P O } } ^ { \mathrm { s i m p l e } }$ , the second terms give the two corrections in equation $^ { 8 8 , }$ , and the last terms form $C$ . The latter use their respective sample targets and are independent of θ because the data and $v _ { \mathrm { o l d } }$ are fixed during differentiation. □

Interpretation. The theorem expresses FlowCPO as $\beta$ times simplified FlowDPO, plus two corrections determined by the distance between $v _ { \theta }$ and $v _ { \mathrm { o l d } }$ and a term $C$ that is constant during each gradient step. $\mathbf { A } \mathbf { t } \beta = 1$ , only the quadratic constraint on dispreferred inputs remains, with coefficient 2. For $0 < \beta < 1$ , including our default $\beta = 0 . 5$ , the preferred correction has a negative coefficient, so the two corrections should not both be described as positive regularizers. The ablation analysis of $\beta$ is in Fig. 9.

Why use EMA for interpolation? We use EMA so that the interpolation reference follows the model being trained. Thm. B.1 shows that the two correction terms depend on $v _ { \theta } \mathrm { ~ - ~ } v _ { \mathrm { o l d } }$ at the corresponding preferred or dispreferred input. These corrections depend on the squared prediction differences, so a reference that falls far behind can substantially change the objective. If we use the frozen $v _ { \mathrm { r e f } }$ for interpolation, the reference stays at initialization while $v _ { \theta }$ changes during fine-tuning. EMA updates the parameters of $v _ { \mathrm { o l d } }$ from recent training iterates, with the aim of keeping this gap small. Thus, $v _ { \mathrm { r e f } }$ provides the fixed in-domain offline data, while $v _ { \mathrm { o l d } }$ follows training and is used to form $\mu _ { \theta }$ and $\nu _ { \theta } .$

Advantages of FlowCPO over FlowDPO. For $\lambda = 1$ , the two losses can be compared directly:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { F l o w D P O } } ^ { \mathrm { s i m p l e } } ( \theta ) = \mathbb { E } \Big [ \| v _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - ( \epsilon - \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } - \| v _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - ( \epsilon - \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } \Big ] , } \\ & { \mathcal { L } _ { \mathrm { F L o w C P O } } ( \theta ) = \mathbb { E } \Big [ \| \mu _ { \theta } ( \mathbf { x } _ { t } ^ { w } , t , c ) - ( \epsilon - \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } + \| \nu _ { \theta } ( \mathbf { x } _ { t } ^ { l } , t , c ) - ( \epsilon - \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{89}
$$

The minus sign allows simplified FlowDPO to decrease without bound if the rejected error grows while the preferred error stays bounded. FLOWCPO adds two nonnegative errors, so its loss is bounded below by zero. This removes one potential source of instability, although a lower bound alone does not guarantee stable training.

## C Connections to Existing Methods Under the Unified Divergence Framework

Sec. 4.3 and Tab. 2 summarize the connections under the unified divergence-based framework.

## C.1 Connection to FlowGRPO

FlowGRPO [32] is an online RL algorithm for flow matching models. Ignoring the KL term added in practice for stabilization, its core objective is

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot | c ) } \big [ r ( \mathbf { x } _ { 0 } , c ) \big ]
$$

By Thm. 4.1, this is exactly equivalent to

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { c } \left[ \mathcal { D } _ { K L } ( \pi _ { \theta } | | \pi ^ { + } ) - \mathcal { D } _ { K L } ( \pi _ { \theta } | | \pi ^ { - } ) \right]
$$

Hence FlowGRPO belongs to the reverse-KL branch of equation 14 with

$$
q _ { \theta } ^ { + } = q _ { \theta } ^ { - } = \pi _ { \theta } , ~ \alpha = 1 , ~ \gamma = - 1 ,
$$

plus the usual stabilization term $\mathcal { D } _ { K L } ( \pi _ { \theta } \Vert \pi _ { \mathrm { r e f } } )$ used in practice.

## C.2 Connection to DiffusionNFT

DiffusionNFT [55] optimizes a supervised loss of the form

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { N F T } } ( \theta ) = \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \mathrm { o l d } } ( \cdot | c ) } \Big [ p ( o = 1 \mid \mathbf { x } _ { 0 } , c ) \| \mu _ { \theta } ( \mathbf { x } _ { t } , t , c ) - u _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad + p ( o = 0 \mid \mathbf { x } _ { 0 } , c ) \| \nu _ { \theta } ( \mathbf { x } _ { t } , t , c ) - u _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{90}
$$

Similar to equation 8, we can have:

$$
\begin{array} { r l } & { \pi _ { \mathrm { o l d } } ( \mathbf { x } _ { 0 } \mid c ) p ( o = 1 \mid \mathbf { x } _ { 0 } , c ) = \pi ^ { + } ( \mathbf { x } _ { 0 } \mid c ) p _ { \pi _ { \mathrm { o l d } } } ( o = 1 \mid c ) } \\ & { \pi _ { \mathrm { o l d } } ( \mathbf { x } _ { 0 } \mid c ) p ( o = 0 \mid \mathbf { x } _ { 0 } , c ) = \pi ^ { - } ( \mathbf { x } _ { 0 } \mid c ) p _ { \pi _ { \mathrm { o l d } } } ( o = 0 \mid c ) } \end{array}
$$

Therefore, with formula equation 34 in Thm. A.7, equation 90 can be rewritten as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { N F T } } ( \theta ) = \mathbb { E } _ { c } \Big [ p _ { \pi _ { \mathrm { o l d } } } \big ( o = 1 \ | \ c \big ) \mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi ^ { + } ( \cdot \vert c ) } \| \mu _ { \theta } ( \mathbf { x } _ { t } , t , c ) - u _ { t } \| _ { 2 } ^ { 2 } \Big ] } \\ & { \qquad + \mathbb { E } _ { c } \Big [ p _ { \pi _ { \mathrm { o l d } } } \big ( o = 0 \ | \ c \big ) \mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi ^ { - } ( \cdot \vert c ) } \| \nu _ { \theta } ( \mathbf { x } _ { t } , t , c ) - u _ { t } \| _ { 2 } ^ { 2 } \Big ] } \\ & { \qquad \quad \geq \mathbb { E } _ { c } \Big [ p _ { \pi _ { \mathrm { o l d } } } \big ( o = 1 \ | \ c \big ) \mathbb { E } _ { \pi ^ { + } } \left[ - \log \pi _ { \theta } ^ { + } ( \mathbf { x } _ { 0 } \ | \ c ) \right] } \\ & { \qquad \quad + p _ { \pi _ { \mathrm { o l d } } } \big ( o = 0 \ | \ c \big ) \mathbb { E } _ { \pi ^ { - } } \left[ - \log \pi _ { \theta } ^ { - } ( \mathbf { x } _ { 0 } \ | \ c ) \right] \Big ] + C } \\ & { \qquad = \mathbb { E } _ { c } \big [ p _ { \pi _ { \mathrm { o l d } } } \big ( o = 1 \ | \ c \big ) D _ { K L } \big ( \pi ^ { + } \| \pi _ { \theta } ^ { + } \big ) + p _ { \pi _ { \mathrm { o l d } } } \big ( o = 0 \ | \ c \big ) D _ { K L } \big ( \pi ^ { - } \| \pi _ { \theta } ^ { - } \big ) \big ] + C } \end{array}
$$

where $C$ is a constant independent of θ. The both sides achieve the same optimal solution when $\mu _ { \theta ^ { \star } } ( \mathbf { x } _ { t } ^ { + } , t , c ) = \mathbb { E } [ \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ^ { + } \mid \mathbf { x } _ { t } ^ { + } , t , c ]$ and $\nu _ { \theta ^ { \star } } ( \mathbf { x } _ { t } ^ { - } , t , c ) = \mathbb { E } [ \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ^ { - } \mid \mathbf { x } _ { t } ^ { - } , t , c ]$

Therefore, DiffusionNFT corresponds to the forward-KL branch of equation 14 with

$$
q _ { \theta } ^ { + } = \pi _ { \theta } ^ { + } , \qquad q _ { \theta } ^ { - } = \pi _ { \theta } ^ { - } , \qquad \alpha = p _ { \pi _ { \mathrm { o l d } } } ( o = 1 \mid c ) , \qquad \gamma = p _ { \pi _ { \mathrm { o l d } } } ( o = 0 \mid c ) .
$$

## C.3 Connection to AWM

AWM [51] aims to optimize the following objective:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot | c ) } \big [ r ( \mathbf { x } _ { 0 } , c ) \big ]\tag{91}
$$

From equation 16, we have:

$$
\begin{array} { r } { - \log \pi _ { \theta } ( x _ { 0 } \mid c ) = H ( q _ { \varepsilon } ^ { x _ { 0 } } ) + \mathbb { E } _ { t , \epsilon } [ w ( t ) \| v _ { \theta } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - x _ { 0 } ) \| ^ { 2 } ] + \mathcal { G } _ { \varepsilon , w } ( \theta ; x _ { 0 } ) + \mathcal { B } _ { \varepsilon } ( \theta ; x _ { 0 } ) } \end{array}\tag{92}
$$

We set $\bar { \theta }$ as the stop gradient of $\theta ,$ we have:

$$
\begin{array} { r } { \log \frac { \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) } { \pi _ { \bar { \theta } } ( \mathbf { x } _ { 0 } \mid c ) } = - \mathbb { E } _ { t , \epsilon } [ w ( t ) \| v _ { \theta } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } ] } \\ { + \mathbb { E } _ { t , \epsilon } [ w ( t ) \| v _ { \bar { \theta } } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } ] } \\ { : = \Delta \mathcal { L } _ { \mathrm { F M } } } \end{array}
$$

Therefore, the objective equation 91 can be rewritten as

$$
\begin{array} { r l } & { \underset { \theta } { \operatorname* { m a x } } \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot | c ) } [ r ( \mathbf { x } _ { 0 } , c ) ] } \\ & { = \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \bar { \theta } } ( \cdot | c ) } \left[ \frac { \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) } { \pi _ { \bar { \theta } } ( \mathbf { x } _ { 0 } \mid c ) } r ( \mathbf { x } _ { 0 } , c ) \right] } \\ & { = \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \bar { \theta } } ( \cdot | c ) } \left[ \exp ( \Delta \mathcal { L } _ { \mathrm { F M } } ) r ( \mathbf { x } _ { 0 } , c ) \right] } \end{array}
$$

Therefore, AWM is optimized over forward process, while its derivation of $q _ { \theta } ^ { + } , q _ { \theta } ^ { - } , \alpha , \gamma$ is the same as FlowGRPO. Thus,

$$
q _ { \theta } ^ { + } = q _ { \theta } ^ { - } = \pi _ { \theta } , ~ \alpha = 1 , ~ \gamma = - 1 ,
$$

Because $\Delta \mathcal { L } _ { \mathrm { F M } } = 0$ due to $\bar { \theta }$ is the stop gradient of θ, we know $\exp ( x ) = 1 + x$ when x is small. Therefore, we can simplify the objective as

$$
\begin{array} { r l } & { \underset { \theta } { \mathrm { m a x } } \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot \vert c ) } [ r ( \mathbf { x } _ { 0 } , c ) ] } \\ & { = \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot \vert c ) } [ \exp ( \Delta \mathcal { L } _ { \mathrm { F M } } ) r ( \mathbf { x } _ { 0 } , c ) ] } \\ & { = \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot \vert c ) } [ ( 1 + \Delta \mathcal { L } _ { \mathrm { F M } } ) r ( \mathbf { x } _ { 0 } , c ) ] } \\ & { = \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot \vert c ) } [ r ( \mathbf { x } _ { 0 } , c ) + \Delta \mathcal { L } _ { \mathrm { F M } } r ( \mathbf { x } _ { 0 } , c ) ] } \\ & { = - \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot \vert c ) } [ r ( \mathbf { x } _ { 0 } , c ) \mathbb { E } _ { t , \epsilon } [ w ( t ) \| v _ { \theta } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } ] ] + C } \\ & { = - \mathbb { E } _ { c , \mathbf { x } _ { 0 } \sim \pi _ { \theta } ( \cdot \vert c ) , t , \epsilon } [ w ( t ) r ( \mathbf { x } _ { 0 } , c ) ] \| v _ { \theta } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } ] + C } \end{array}
$$

## C.4 Connection to RAM

First, let’s remind the objective of RAM [3]:

$$
\mathcal { L } _ { \mathrm { R A M } } ( \theta ) = \mathbb { E } _ { c , t } [ \| v _ { \theta } ( \mathbf { x } _ { t } , t , c ) - s g ( v _ { \mathrm { r e f } } ( \mathbf { x } _ { t } , t , c ) + r ( \mathbf { x } _ { 0 } , c ) ( ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) - v _ { \theta } ( \mathbf { x } _ { t } , t , c ) ) ) \| ^ { 2 } ]
$$

where $s g$ is the stop gradient operator.

Let us define a new objective:

$$
\begin{array} { r l } & { \qquad \mathcal { L } _ { 2 } ( \theta ) = \mathbb { E } _ { c , t } [ r ( { \mathbf x } _ { 0 } , c ) ] \| v _ { \theta } ( { \mathbf x } _ { t } , t , c ) - ( { \mathbf x } _ { 1 } - { \mathbf x } _ { 0 } ) \| ^ { 2 } ] + \mathbb { E } _ { c , t } [ \| v _ { \theta } ( { \mathbf x } _ { t } , t , c ) - v _ { \mathrm { r e f } } ( { \mathbf x } _ { t } , t , c ) \| ^ { 2 } ] } \\ & { { \mathrm { I t i s ~ e a s y ~ t o ~ c h e c k ~ t h a t ~ } } \nabla _ { \theta } \mathcal { L } _ { 2 } ( \theta ) = \nabla _ { \theta } \mathcal { L } _ { \mathrm { R A M } } ( \theta ) . } \end{array}
$$

Therefore, RAM can be considered as AWM objective plus a reference term.

Thus, it is also optimized over forward process, while its derivation of $q _ { \theta } ^ { + } , q _ { \theta } ^ { - } , \alpha .$ , γ is the same as FlowGRPO. Thus,

$$
q _ { \theta } ^ { + } = q _ { \theta } ^ { - } = \pi _ { \theta } , ~ \alpha = 1 , ~ \gamma = - 1 ,
$$

## C.5 Connection to SFT

The reference distribution decomposes as a mixture of the positive and negative posteriors.

Lemma C.1. For every context c,

$$
\pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } \mid c ) = p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) \pi ^ { + } ( \mathbf { x } _ { 0 } \mid c ) + p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) \pi ^ { - } ( \mathbf { x } _ { 0 } \mid c ) .
$$

Proof. Multiplying the definitions in equation 8 by the corresponding normalizing constants and summing the two identities gives

$$
\begin{array} { r l } & { p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) \pi ^ { + } ( \mathbf { x } _ { 0 } \mid c ) + p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) \pi ^ { - } ( \mathbf { x } _ { 0 } \mid c ) } \\ & { ~ = \pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } \mid c ) [ p ( o = 1 \mid \mathbf { x } _ { 0 } , c ) + p ( o = 0 \mid \mathbf { x } _ { 0 } , c ) ] } \\ & { ~ = \pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } \mid c ) . } \end{array}
$$

Standard SFT maximizes the log-likelihood of samples from $\pi _ { \mathrm { r e f } }$ . By Thm. A.7, we know that when $\mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi _ { \mathrm { r e f } } ( \cdot | c ) } \big \lceil \big \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \big \| ^ { 2 } \big \rceil$ achieves the minimum, $\mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi _ { \mathrm { r e f } } ( \cdot | c ) } \big [ - \log \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) \big ]$ also achieves the minimum. Therefore, we have: By Lem. C.1,

$$
\begin{array} { l } { { \displaystyle \ = \arg \operatorname* { m i n } _ { \theta } \mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi _ { \mathrm { r e f } } ( \cdot \vert c ) } [ - \log \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) ] } } \\ { { \displaystyle \ = \int \pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } \mid c ) [ - \log \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) ] d \mathbf { x } _ { 0 } } } \\ { { \displaystyle \ = \int ( p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) \pi ^ { + } ( \mathbf { x } _ { 0 } \mid c ) + p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) \pi ^ { - } ( \mathbf { x } _ { 0 } \mid c ) ) [ - \log \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) ] d \mathbf { x } _ { 0 } } } \\ { { \displaystyle = p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) \mathcal { D } _ { K L } ( \pi ^ { + } \| \pi _ { \theta } ) + p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) \mathcal { D } _ { K L } ( \pi ^ { - } \| \pi _ { \theta } ) } } \end{array}
$$

Thus SFT corresponds to the forward-KL branch of equation 14 with

$$
q _ { \theta } ^ { + } = q _ { \theta } ^ { - } = \pi _ { \theta } , \qquad \alpha = p _ { \pi _ { \mathrm { r e f } } } ( o = 1 \mid c ) , \qquad \gamma = p _ { \pi _ { \mathrm { r e f } } } ( o = 0 \mid c ) .
$$

## C.6 Connection to RFT

RFT keeps only preferred samples. By Thm. A.7, we know that when $\mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi _ { \mathrm { r e f } } ( \cdot | c ) } [ \| v _ { \theta } ( \mathbf { x } _ { t } , t ) - $ $\left( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \right) \parallel ^ { 2 } ]$ achieves the minimum, $\mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi _ { \mathrm { r e f } } ( \cdot | c ) } \big [ - \log \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) \big ]$ also achieves the minimum. Therefore, we have:

$$
\begin{array} { r l } { \mathbf { \Pi } } & { = \arg \underset { \theta } { \operatorname* { m i n } } \mathbb { E } _ { \mathbf { x } _ { 0 } \sim \pi ^ { + } ( \cdot | c ) } [ - \log \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) ] } \\ & { = \arg \underset { \theta } { \operatorname* { m i n } } \mathcal { D } _ { K L } ( \pi ^ { + } \| \pi _ { \theta } ) } \end{array}
$$

Therefore RFT corresponds to the forward-KL branch of equation 14 with

$$
q _ { \theta } ^ { + } = \pi _ { \theta } , \qquad \alpha = 1 , \qquad \gamma = 0 .
$$

## C.7 Connection to FlowDPO

The connection to FlowDPO [33] is heuristic rather than formal. Starting from the KL-regularized RLHF objective, DPO uses the reward parameterization

$$
\begin{array} { r } { r ( \mathbf { x } _ { 0 } , c ) = \eta \log \frac { \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) } { \pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } \mid c ) } + \eta \log Z ( c ) , } \end{array}\tag{93}
$$

which, when substituted into the Bradley–Terry preference likelihood, yields the usual DPO objective

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \theta ) = \mathbb { E } _ { c , \mathbf { x } _ { 0 } ^ { w } , \mathbf { x } _ { 0 } ^ { l } } \left[ - \log \sigma \left( \eta \log \frac { \pi _ { \theta } ( \mathbf { x } _ { 0 } ^ { w } \mid c ) } { \pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } ^ { w } \mid c ) } - \eta \log \frac { \pi _ { \theta } ( \mathbf { x } _ { 0 } ^ { l } \mid c ) } { \pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } ^ { l } \mid c ) } \right) \right] } \end{array}
$$

From equation 16, we have:

$$
- \log \pi _ { \theta } ( x _ { 0 } \mid c ) = H ( q _ { \varepsilon } ^ { x _ { 0 } } ) + \mathbb { E } _ { t , \epsilon } [ w ( t ) \| v _ { \theta } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - x _ { 0 } ) \| ^ { 2 } ] + \mathcal { G } _ { \varepsilon , w } ( \theta ; x _ { 0 } ) + \mathcal { B } _ { \varepsilon } ( \theta ; x _ { 0 } )
$$

Therefore, we have:

$$
\begin{array} { r } { \log \frac { \pi _ { \theta } ( \mathbf { x } _ { 0 } \mid c ) } { \pi _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } \mid c ) } = \Delta \mathcal { L } _ { \mathrm { F M } } ( \mathbf { x } _ { 0 } ) + \Delta \mathcal { G } _ { \varepsilon , w } ( \mathbf { x } _ { 0 } ) + \Delta \mathcal { B } _ { \varepsilon } ( \mathbf { x } _ { 0 } ) } \end{array}\tag{94}
$$

where

$$
\begin{array} { r l } & { \Delta \mathcal { L } _ { \mathrm { F M } } ( \mathbf { x } _ { 0 } ) : = \mathbb { E } _ { t , \epsilon } [ - w ( t ) \| v _ { \theta } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } + w ( t ) \| v _ { \mathrm { r e f } } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \| ^ { 2 } ] } \\ & { \Delta \mathcal { G } _ { \epsilon , w } ( \mathbf { x } _ { 0 } ) : = - \mathcal { G } _ { \epsilon , w } ( \theta ; \mathbf { x } _ { 0 } ) + \mathcal { G } _ { \epsilon , w } ( \mathrm { r e f } ; \mathbf { x } _ { 0 } ) } \\ & { \Delta B _ { \epsilon } ( \mathbf { x } _ { 0 } ) : = - B _ { \epsilon } ( \theta ; \mathbf { x } _ { 0 } ) + B _ { \epsilon } ( \mathrm { r e f } ; \mathbf { x } _ { 0 } ) } \end{array}
$$

Therefore, we can simplify the DPO objective in the following:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { T p S } } ( \theta ) = \mathbb { E } _ { \epsilon _ { \mathrm { A X } ^ { \prime } } , \epsilon _ { \mathrm { X } } ^ { \prime } } \big [ - \log \sigma \big ( | \hat { \mathcal { X } } \Delta _ { \mathcal { F } , \mathrm { N I } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) + \Delta \hat { \mathcal { L } } _ { \mathrm { S } , \mathrm { t r } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) + \Delta \hat { \mathcal { L } } _ { \mathrm { S } , \mathrm { t r } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) } \\ & { \qquad - \Delta \hat { \mathcal { L } } _ { \mathrm { S } , \mathrm { t r } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) - \Delta \hat { \mathcal { L } } _ { \mathrm { S } , \mathrm { t r } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) - \Delta \hat { \mathcal { L } } _ { \mathrm { S } , \mathrm { t r } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) \big ) \big ] } \\ & { = \mathbb { E } _ { \epsilon _ { \mathrm { A X } ^ { \prime } } , \epsilon _ { \mathrm { W } } ^ { \prime } } \big [ - \log \sigma \big ( | \big ( \Delta \mathcal { L } _ { \mathrm { F } , \mathrm { M X } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) - \Delta \hat { \mathcal { L } } _ { \mathrm { Y } , \mathrm { W } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) \big ) } \\ &  \qquad + \eta \big ( \Delta \mathcal { L } _ { \mathrm { Z } , \mathrm { t r } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) + \Delta \hat { \mathcal { L } } _ { \mathrm { S } , \mathrm { t r } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) - \Delta \hat { \mathcal { L } } _ { \mathrm { Z } , \mathrm { u e } } ( \mathbf { x } _ { \mathrm { W } } ^ { \prime } ) - \Delta \hat \end{array}
$$

where

$$
\delta _ { \theta } ( \mathbf { x } _ { 0 } ) : = \Vert v _ { \theta } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \Vert ^ { 2 } - \Vert v _ { \mathrm { r e f } } ( \mathbf { x } _ { t } , t , c ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ) \Vert ^ { 2 } .
$$

$$
\mathcal { L } _ { \mathrm { F l o w D P O } } ( \theta ) = \mathbb { E } _ { c , \mathbf { x } _ { 0 } ^ { w } , \mathbf { x } _ { 0 } ^ { l } , t , \epsilon } \big [ - \log \sigma \big ( 2 \eta w ( t ) [ \delta _ { \theta } ( \mathbf { x } _ { 0 } ^ { l } ) - \delta _ { \theta } ( \mathbf { x } _ { 0 } ^ { w } ) ] \big ) \big ] .
$$

Therefore, the FlowDPO objective plus a residual term is an upper bound of the DPO objective.

Algorithm 1 Flow Contrastive Preference Optimization (FlowCPO)   
Require: Pretrained $v _ { \mathrm { r e f } } .$ , dataset ${ \mathcal D } = \{ ( c , { \bf x } _ { 0 } ^ { w } , { \bf x } _ { 0 } ^ { l } ) \}$ , flow interpolation coefficient $\beta ,$ negative regularization   
weight λ, EMA coefficient $\eta ,$ learning rate ι.   
1: Init: $v _ { \theta }  v _ { \mathrm { r e f } } , v _ { \mathrm { o l d } }  v _ { \mathrm { r e f } } .$   
2: for each training iteration do   
3: Sample batch $( c , \mathbf { x } _ { 0 } ^ { w } , \mathbf { x } _ { 0 } ^ { l } ) \sim \mathcal { D } , t \sim \mathcal { U } ( 0 , 1 ) , \epsilon \sim \mathcal { N } ( 0 , I ) .$   
4: Forward: $\mathbf { x } _ { t } ^ { \{ \dot { w } , l \} } = ( 1 - t ) \cdot \mathbf { x } _ { 0 } ^ { \{ w , l \} } + t \cdot \epsilon ; \quad \mathrm { T a r g e t } u _ { t } ^ { \{ \dot { w } , l \} } = \epsilon - \mathbf { x } _ { 0 } ^ { \{ w , l \} } .$   
5: Flow Mixing:   
$\mu _ { \theta } = ( 1 - \bar { \beta } ) v _ { \mathrm { o l d } } + \beta v _ { \theta } , \quad \nu _ { \theta } = ( 1 + \beta ) v _ { \mathrm { o l d } } - \beta v _ { \theta } .$   
6: Loss Update:   
$\mathcal { L } = \| \dot { \mu } _ { \theta } ( \mathbf { x } _ { t } ^ { w } , c , t ) - u _ { t } ( \mathbf { x } _ { t } ^ { w } | \mathbf { x } _ { 0 } ^ { w } ) \| _ { 2 } ^ { 2 } + \lambda \| \nu _ { \theta } ( \mathbf { x } _ { t } ^ { l } , c , t ) - u _ { t } ( \mathbf { x } _ { t } ^ { l } | \mathbf { x } _ { 0 } ^ { l } ) \| _ { 2 } ^ { 2 }$   
7: $\begin{array} { r } { \theta  \stackrel { \cdots } { \theta } - \stackrel { \cdot } { \iota } \cdot \nabla _ { \theta } \mathcal { L } . } \end{array}$   
8: $v _ { \mathrm { o l d } }  E M A _ { \eta } ( v _ { \theta } , v _ { \mathrm { o l d } } )$   
9: end for   
10: return v<sub>θ</sub>

## D Experimental Details

## D.1 Experimental Details of Sec. 5.2

In-Domain Offline Preference Data. For the in-domain offline training setting, we build a static preference dataset from a frozen copy of the pretrained reference model Stable Diffusion 3.5 Medium (SD3.5-M). We sample prompts from the three target sources: the GenEval prompt set, the OCR prompt set, and the general-preference prompt set. The resulting in-domain prompt pools contain 50,000 training prompts for GenEval, 19,653 training prompts for OCR, and 25,432 training prompts for general preference. Their held-out evaluation splits contain 2212, 1018, and 2048 prompts, respectively. For each prompt, we generate 16 candidate images with the frozen reference model and convert them into a preferred/dispreferred pair using a fixed offline pipeline: the highest-scoring sample becomes the preferred and the lowest-scoring sample becomes the dispreferred. The GenEval and OCR training data use single-metric filtering aligned with their target benchmark. The generalpreference training data uses an equal-weight multi-reward score with PickScore:HPS v2.1:CLIP Score = 1:1:1 during preferred/dispreferred construction. Because the generator parameters stay fixed during data construction, the resulting training distribution remains stationary throughout optimization. These preferred/dispreferred pairs should be read as empirical source distributions induced by the fixed best-of-16 pipeline, rather than as exact samples from the idealized posteriors $\pi ^ { + }$ and $\pi ^ { - }$ introduced in the theory.

All images used for data construction and evaluation are generated at a resolution of $5 1 2 \times 5 1 2$ with 40 inference steps using the ODE sampler. Unless otherwise noted, we use a guidance scale of 4.5 during dataset construction. We leave negative prompts blank. The VAE and text encoder are frozen, and training prompts do not overlap with validation prompts or evaluation benchmarks. The complete reported experiment suite used approximately 288 aggregate A100 GPU-hours.

Out-of-Domain Offline Preference Data. For the out-of-domain offline training setting, we use Open Image Preferences v1 Results from the Data Is Better Together collection on Hugging Face [7]. This dataset contains roughly 10K text-to-image preference pairs with community annotations, and the images are generated by open models including FLUX.1-Dev and SD3.5-Large. We use it to evaluate the out-of-domain regime because the training samples are not generated by the reference model used in our fine-tuning experiments.

Training Configuration. We fine-tune SD3.5-M with LoRA on 8×A100 GPUs in fp16 precision. Unless otherwise specified, the LoRA hyperparameters are $r = 3 2$ and $\alpha = 6 4$ . We optimize with AdamW using a learning rate of 3e-4, cosine decay, and weight decay of 1e-4. The global batch size is 16. The EMA decay η is 0.99, the default value of λ is 1, and reported checkpoints are selected with the corresponding validation score for each training target.

Checkpoint Selection. All fine-tuning runs are monitored for 2K training steps. For the in-domain specialists, we report the checkpoint with the best task-aligned validation score within that 2K-step budget: the GenEval model is selected by GenEval validation score, the OCR model by OCR validation score, and the general-preference model by the equal-weight composite validation score over PickScore, HPS v2.1, and CLIP Score. For the OOD setting, where one model is shared across all downstream evaluations, we select the checkpoint with the highest validation PickScore within the same 2K-step budget. We apply the same protocol to FlowCPO and the offline baselines in the corresponding setting.

## D.2 Experimental Results on Optimization Targets

We report results for three capabilities: semantic alignment, typographic generation, and general preference alignment. Tab. 5, Tab. 6, and Tab. 7 correspond to in-domain fine-tuning with domainspecific reward filtering, whereas Tab. 8 reports the out-of-domain setting. The first three tables therefore summarize separate in-domain specialist models, while the OOD table evaluates single models trained once on out-of-domain data. Unless otherwise noted, each fine-tuned SD3.5-M entry is the mean over five independent training and evaluation runs with different random seeds, and the uncertainty is the sample standard deviation of the five run-level metric values. Pretrained baselines are shown as single evaluations for reference.

Tab. 5 reports quantitative results on GenEval, with representative qualitative comparisons in Fig. 3. Tab. 6 summarizes typographic generation performance measured by OCR accuracy, with examples in Fig. 4. Tab. 7 presents the in-domain results for general preference alignment; because training data in that setting are filtered with PickScore, CLIP Score, and HPS v2.1, the remaining metrics are the more informative check of transfer beyond the filtering pipeline. Tab. 8 reports the out-of-domain setting, with qualitative examples in Fig. 6, Fig. 7, and Fig. 8. Unless otherwise noted, we use FlowDPO temperature 100 and FlowCPO $\beta = 0 . 5$ as the default settings.

Table 5: Quantitative comparison of fine-tuning SD3.5-M using different methods under the in-domain offline training setting. The fine-tuning data are filtered solely by the GenEval reward function. We evaluate each model with several Classifier-Free Guidance (CFG) scales. The broad set of auxiliary metrics is included to monitor whether optimizing only for GenEval degrades other capabilities. Fine-tuned SD3.5-M variants are reported as mean ± standard deviation over 5 runs; pretrained baselines are listed as single evaluations. Within the SD3.5-M group, the best results are highlighted in bold, and the second best are underlined.
<table><tr><td>Model</td><td>CFG</td><td>GenEval</td><td>OCR</td><td>PickScore</td><td>ClipScore</td><td>HPSv2.1</td><td>Aesthetic</td><td>ImgRwd</td><td>UniRwd</td></tr><tr><td>Pretrained Model Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SD-XL</td><td></td><td>0.55</td><td>0.14</td><td>22.42</td><td>0.287</td><td>0.280</td><td>5.60</td><td>0.76</td><td>2.93</td></tr><tr><td>SD3.5-L</td><td></td><td>0.71</td><td>0.68</td><td>22.91</td><td>0.289</td><td>0.288</td><td>5.50</td><td>0.96</td><td>3.25</td></tr><tr><td>FLUX.1-Dev</td><td></td><td>0.66</td><td>0.59</td><td>22.84</td><td>0.295</td><td>0.274</td><td>5.71</td><td>0.96</td><td>3.27</td></tr><tr><td>SD3.5-M Fine-Tuning (filtered by GenEval)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td></tr><tr><td>Base Model</td><td>1.0 3.0</td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td></td><td></td><td>0.59</td><td>0.47</td><td>22.28</td><td>0.287</td><td>0.284</td><td>5.38</td><td>0.71</td><td>2.96</td></tr><tr><td></td><td>4.5</td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td></td><td>1.0</td><td>0.5883 ± 0.0100</td><td>0.1421 ± 0.0033</td><td>21.6937 ± 0.0125</td><td>0.2735 ± 0.0009</td><td>0.2686 ± 0.0005</td><td>5.3183 ± 0.0155</td><td>0.4381 ± 0.0179</td><td>2.6175 ± 0.0135</td></tr><tr><td>+ RFT [49, 5]</td><td>3.0</td><td>0.7423 ± 0.0097</td><td>0.5003 ± 0.0070</td><td>22.4661 ± 0.0212</td><td>0.2941 ± 0.0006</td><td>0.3010 ± 0.0006</td><td>5.3998 ± 0.0092</td><td>1.0200 ± 0.0172</td><td>3.1360 ± 0.0145</td></tr><tr><td></td><td>4.5</td><td>0.7489 ± 0.0045</td><td>0.5562 ± 0.0098</td><td>22.4864 ± 0.0099</td><td>0.2966 ± 0.0005</td><td>0.3031 ± 0.0002</td><td> $\frac { 1 . 0 3 9 9 0 \pm 0 . 0 0 3 2 } { 5 . 4 0 7 0 \pm 0 . 0 0 5 0 }$ </td><td> $1 . 0 8 0 8 \pm 0 . 0 0 7 4$ </td><td>3.1615 ± 0.0080</td></tr><tr><td></td><td>1.0</td><td>0.5892 ± 0.0046</td><td> $\overline { { 0 . 2 3 2 7 \pm 0 . 0 0 5 3 } }$ </td><td>21.4067 ± 0.0324</td><td> $0 . 2 6 9 6 \pm 0 . 0 0 1 0$ </td><td> $\overline { { 0 . 2 5 6 2 \pm 0 . 0 0 0 7 } }$ </td><td> $\overline { { 5 . 2 8 1 7 \pm 0 . 0 0 5 8 } }$ </td><td> $\overline { { 0 . 4 5 6 0 \pm 0 . 0 2 2 5 } }$ </td><td>2.5870 ± 0.0205</td></tr><tr><td>+ FlowDPO [33]</td><td>3.0</td><td>0.8118 ± 0.0044</td><td>0.4534 ± 0.0075</td><td>22.3095 ± 0.0182</td><td>0.2970 ± 0.0006</td><td>0.2938 ± 0.0009</td><td>5.3570 ± 0.0112</td><td>1.0811 ± 0.0094</td><td>3.1615 ± 0.0200</td></tr><tr><td></td><td>4.5</td><td>0.8107 ± 0.0046</td><td> $0 . 5 0 6 7 \pm 0 . 0 0 7 4$ </td><td>22.3091 ± 0.0138</td><td>0.2979 ± 0.0005</td><td>0.2963 ± 0.0003</td><td>5.3479 ± 0.0052</td><td>1.1256 ± 0.0078</td><td>3.1680 ± 0.0060</td></tr><tr><td></td><td>1.0</td><td>0.7596 ± 0.0106</td><td>0.2535 ± 0.0098</td><td> $\overline { { 2 1 . 8 0 4 5 \pm 0 . 0 1 5 7 } }$ </td><td>0.2804 ± 0.0001</td><td>0.2673 ± 0.0008</td><td> $\overline { { 5 . 2 4 8 7 \pm 0 . 0 1 3 9 } }$ </td><td>0.6231 ± 0.0160</td><td> $\overline { { 2 . 8 0 4 0 \pm 0 . 0 1 2 5 } }$ </td></tr><tr><td>+ FlowCPO (β = 0.5, Ours)</td><td>3.0</td><td>0.8415 ± 0.0036</td><td>0.5203 ± 0.0055</td><td>22.1952 ± 0.0184</td><td>0.2957 ± 0.0005</td><td>0.2885 ± 0.0007</td><td> $5 . 2 6 6 2 \pm 0 . 0 0 8 5$ </td><td> $0 . 9 9 9 9 \pm 0 . 0 0 7 6$ </td><td> $3 . 1 1 7 0 \pm 0 . 0 0 8 5$ </td></tr><tr><td></td><td>4.5</td><td>0.8161 ± 0.0052</td><td> $0 . 5 7 0 2 \pm 0 . 0 0 8 2$ </td><td>21.9438 ± 0.0086</td><td> $0 . 2 9 3 3 \pm 0 . 0 0 0 5$ </td><td> $0 . 2 8 0 7 \pm 0 . 0 0 0 4$ </td><td> $5 . 1 \dot { 6 } 1 0 \pm 0 . 0 0 6 2$ </td><td>0.9184 ± 0.0094</td><td> $\dot { 3 } . 0 2 4 5 \pm 0 . 0 1 5 0$ </td></tr></table>

Table 6: Quantitative comparison of fine-tuning SD3.5-M using different methods (RFT, FlowDPO, FlowCPO) under the in-domain offline training setting. The fine-tuning data are filtered solely by the OCR reward metric. We evaluate the models with several Classifier-Free Guidance (CFG) scales. The broad set of auxiliary metrics is included to monitor whether optimizing only for OCR degrades other generation capabilities. Fine-tuned SD3.5-M variants are reported as mean ± standard deviation over 5 runs; pretrained baselines are listed as single evaluations. Within the SD3.5-M fine-tuning group, the best results are highlighted in bold, and the second best are underlined.
<table><tr><td>Model</td><td>CFG</td><td>GenEval</td><td>OCR</td><td>PickScore</td><td>ClipScore</td><td>HPSv2.1</td><td>Aesthetic</td><td>ImgRwd</td><td>UniRwd</td></tr><tr><td>Pretrained Model Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SD-XL</td><td></td><td>0.55</td><td>0.14</td><td>22.42</td><td>0.287</td><td>0.280</td><td>5.60</td><td>0.76</td><td>2.93</td></tr><tr><td>SD3.5-L</td><td></td><td>0.71</td><td>0.68</td><td>22.91</td><td>0.289</td><td>0.288</td><td>5.50</td><td>0.96</td><td>3.25</td></tr><tr><td>FLUX.1-Dev</td><td></td><td>0.66</td><td>0.59</td><td>22.84</td><td>0.295</td><td>0.274</td><td>5.71</td><td>0.96</td><td>3.27</td></tr><tr><td>SD3.5-M Fine-Tuning (filtered by OCR)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1.0</td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td>Base Model</td><td>3.0</td><td>0.59</td><td>0.47</td><td>22.28</td><td>0.287</td><td>0.284</td><td>5.38</td><td>0.71</td><td>2.96</td></tr><tr><td></td><td>4.5</td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td></td><td>1.0</td><td>0.5127 ± 0.0108</td><td>0.3530 ± 0.0057</td><td>21.8831 ± 0.0192</td><td>0.2784 ± 0.0007</td><td>0.2741 ± 0.0005</td><td>5.3589 ± 0.0096</td><td>0.5248 ± 0.0191</td><td>2.7170 ± 0.0160</td></tr><tr><td>+ RFT [49, 5]</td><td>3.0</td><td>0.6577 ± 0.0106</td><td>0.6981 ± 0.0105</td><td>22.5177 ± 0.0194</td><td>0.2964 ± 0.0008</td><td>0.3008 ± 0.0003</td><td>5.3954 ± 0.0043</td><td>1.0491 ± 0.0072</td><td>3.1630 ± 0.0200</td></tr><tr><td></td><td>4.5</td><td>0.6698 ± 0.0101</td><td>0.7202 ± 0.0083</td><td> $2 2 . 5 0 9 8 \pm 0 . 0 1 5 7$ </td><td>0.2979 ± 0.0009</td><td>0.3023 ± 0.0005</td><td>5.3927 ± 0.0068</td><td>1.0816 ± 0.0100</td><td>3.1605 ± 0.0175</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1.0</td><td>0.3140 ± 0.0051</td><td>0.5136 ± 0.0050</td><td> $2 0 . 9 1 7 1 \pm 0 . 0 0 9 0$ </td><td> $0 . 2 5 6 5 \pm 0 . 0 0 0 8$ </td><td> $0 . 2 2 1 9 \pm 0 . 0 0 0 6$ </td><td> $5 . 1 2 4 6 \pm 0 . 0 1 2 7$ </td><td>−0.3015 ± 0.0273</td><td>2.2580 ± 0.0175</td></tr><tr><td>+ FlowDPO [33]</td><td>3.0</td><td>0.5848 ± 0.0098 0.6149 ± 0.0153</td><td>0.7389 ± 0.0096 0.7476 ± 0.0041</td><td>22.3210 ± 0.0104  $2 2 . 4 2 4 9 \pm 0 . 0 1 4 4$ </td><td>0.2927 ± 0.0004 0.2960 ± 0.0006</td><td>0.2873 ± 0.0005  $0 . 2 9 4 3 \pm 0 . 0 0 0 5$ </td><td>5.3620 ± 0.0025 5.3795 ± 0.0064</td><td>0.8559 ± 0.0143 0.9493 ± 0.0182</td><td>3.0315 ± 0.0220</td></tr><tr><td></td><td>4.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3.0955 ± 0.0195</td></tr><tr><td></td><td>1.0</td><td>0.4023 ± 0.0062</td><td> $\overline { { 0 . 8 3 4 9 \pm 0 . 0 0 3 3 } }$  0.8737±0.0034</td><td> $\overline { { 2 1 . 2 1 7 8 \pm 0 . 0 1 8 5 } }$ </td><td> $0 . 2 6 8 9 \pm 0 . 0 0 0 6$ </td><td> $0 . 2 4 3 0 \pm 0 . 0 0 1 1$ </td><td> $5 . 1 7 7 0 \pm 0 . 0 1 0 2$ </td><td>0.0952 ± 0.0182</td><td>2.4955 ± 0.0185</td></tr><tr><td> $+ \mathrm { F l o w C P O } \left( \beta = 0 . 5 , 0 \mathrm { u r s } \right)$ </td><td>3.0 4.5</td><td>0.5962 ± 0.0113  $0 . 6 1 2 4 \pm 0 . 0 1 0 2$ </td><td> $0 . 8 5 8 8 \pm 0 . 0 0 7 8$ </td><td>22.3139 ± 0.0090  $2 2 . 3 6 5 6 \pm 0 . 0 1 5 4$ </td><td>0.2930 ± 0.0002  $0 . 2 9 5 3 \pm 0 . 0 0 0 8$ </td><td>0.2943 ± 0.0005  $0 . 2 9 8 2 \pm 0 . 0 0 0 5$ </td><td>5.3506 ± 0.0079  $5 . 3 5 9 1 \pm 0 . 0 0 4 8$ </td><td>0.9560 ± 0.0115 1.0438 ± 0.0048</td><td>3.0760 ± 0.0085  $3 . 1 1 6 0 \pm 0 . 0 0 3 5$ </td></tr></table>

Table 7: Quantitative comparison of fine-tuning SD3.5-M using different methods (RFT, FlowDPO, and FlowCPO) under the in-domain offline training setting. The fine-tuning data are filtered by a combination of multiple metrics (PickScore, CLIP Score, and HPSv2.1), which are highlighted in gray. We evaluate the models with several Classifier-Free Guidance (CFG) scales. Because the shaded metrics are directly coupled to the filtering pipeline, the remaining metrics are especially useful for assessing transfer beyond the optimizationaligned rewards. Fine-tuned SD3.5-M variants are reported as mean ± standard deviation over 5 runs; pretrained baselines are listed as single evaluations. Within the SD3.5-M fine-tuning group, the best results are highlighted in bold, and the second best are underlined.
<table><tr><td>Model</td><td>CFG</td><td>GenEval</td><td>OCR</td><td>PickScore</td><td>ClipScore</td><td>HPSv2.1</td><td>Aesthetic</td><td>ImgRwd</td><td>UniRwd</td></tr><tr><td>Pretrained Model Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SD-XL</td><td></td><td>0.55</td><td>0.14</td><td>22.42</td><td>0.287</td><td>0.280</td><td>5.60</td><td>0.76</td><td>2.93</td></tr><tr><td>SD3.5-L</td><td></td><td>0.71</td><td>0.68</td><td>22.91</td><td>0.289</td><td>0.288</td><td>5.50</td><td>0.96</td><td>3.25</td></tr><tr><td>FLUX.1-Dev</td><td></td><td>0.66</td><td>0.59</td><td>22.84</td><td>0.295</td><td>0.274</td><td>5.71</td><td>0.96</td><td>3.27</td></tr><tr><td>SD3.5-M Fine-Tuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1.0</td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td>Base Model</td><td>3.0</td><td>0.59</td><td>0.47</td><td>22.28</td><td>0.287</td><td>0.284</td><td>5.38</td><td>0.71</td><td>2.96</td></tr><tr><td></td><td>4.5</td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td></td><td>1.0</td><td>0.5312 ± 0.0111</td><td>0.2351 ± 0.0045</td><td>21.9056 ± 0.0124</td><td>0.2793 ± 0.0002</td><td>0.2759 ± 0.0008</td><td>5.3507 ± 0.0111</td><td>0.5694 ± 0.0213</td><td>2.6590 ± 0.0065</td></tr><tr><td>+ RFT [49, 5]</td><td></td><td>0.6894 ± 0.0086</td><td>0.5722 ± 0.0045</td><td>22.5975 ± 0.0069</td><td>0.2963 ± 0.0007</td><td>0.3041 ± 0.0005</td><td>5.4139 ± 0.0093</td><td>1.0616 ± 0.0096</td><td>3.1370 ± 0.0100</td></tr><tr><td></td><td>3.0 4.5</td><td>0.6974 ± 0.0056</td><td> $\mathbf { 0 . 6 1 8 0 \mathop { = } 0 . 0 0 9 7 }$ </td><td>22.5664 ± 0.0104</td><td>0.2972 ± 0.0009</td><td>0.3054 ± 0.0005</td><td> $5 . 4 2 0 7 \pm 0 . 0 0 6 1$ </td><td> $1 . 1 0 6 8 \pm 0 . 0 0 6 3$ </td><td>3.1520 ± 0.0020</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1.0</td><td> $\overline { { 0 . 0 3 6 3 \pm 0 . 0 1 1 7 } }$  0.3903 ± 0.0100</td><td> $0 . 2 1 3 2 \pm 0 . 0 0 6 1$  0.5549 ± 0.0072</td><td>20.7158 ± 0.0179</td><td> $\smash { 0 . 2 3 9 9 \pm 0 . 0 0 0 6 }$ </td><td> $0 . 2 2 0 5 \pm 0 . 0 0 1 0$ </td><td> $\smash { 5 . 2 2 0 2 \pm 0 . 0 1 1 3 }$ </td><td> $- 0 . 4 6 1 0 \pm 0 . 0 2 2 2$ </td><td> $\overline { { 2 . 1 0 7 5 \pm 0 . 0 1 2 0 } }$ </td></tr><tr><td>+ FlowDPO [33]</td><td>3.0</td><td> $0 . 4 6 1 0 \pm 0 . 0 1 3 7$ </td><td> $0 . 5 7 9 1 \pm 0 . 0 0 7 1$ </td><td>22.7637 ± 0.0209 22.8912 ± 0.0192</td><td>0.2974 ± 0.0008 0.3010 ± 0.0008</td><td>0.3014 ± 0.0007 0.3107 ± 0.0006</td><td>5.5590 ± 0.0089 5.5562 ± 0.0107</td><td>1.0403 ± 0.0089  $1 . 1 5 4 3 \pm 0 . 0 0 7 2$ </td><td>3.0915 ± 0.0195</td></tr><tr><td></td><td>4.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $3 . 1 8 4 0 \pm 0 . 0 1 9 0$ </td></tr><tr><td></td><td>1.0</td><td> $\overline { { 0 . 5 3 2 5 \pm 0 . 0 0 3 5 } }$ </td><td> $\overline { { 0 . 3 4 2 6 \pm 0 . 0 0 5 3 } }$ </td><td> $\overline { { 2 2 . 4 8 0 5 \pm 0 . 0 1 5 5 } }$ </td><td> $\overline { { 0 . 2 7 9 6 \pm 0 . 0 0 0 7 } }$ </td><td> $\overline { { 0 . 2 9 0 0 \pm 0 . 0 0 0 3 } }$ </td><td> $\overline { { { \bf 5 . 6 3 9 2 } \pm 0 . 0 1 0 0 } }$ </td><td> $\overline { { 0 . 9 0 1 4 \pm 0 . 0 1 3 9 } }$ </td><td> $\overline { { 2 . 9 3 0 0 \pm 0 . 0 2 0 0 } }$ </td></tr><tr><td>+ FlowCPO (β = 0.5, Ours)</td><td>3.0</td><td>0.6487 ± 0.0088</td><td>0.5888 ± 0.0096</td><td>22.9432 ± 0.0164</td><td>0.2994 ± 0.0005</td><td>0.3114±0.0004</td><td> $5 . 5 3 9 5 \pm 0 . 0 0 6 5$ </td><td>1.2480 ± 0.0071</td><td>3.3020 ± 0.0110</td></tr><tr><td></td><td>4.5</td><td> $0 . 6 5 2 6 \pm 0 . 0 0 7 9$ </td><td> $\frac { \phantom { - } } { 0 . 5 7 6 2 \pm 0 . 0 1 3 6 }$ </td><td> $2 2 . 6 7 2 5 \pm 0 . 0 0 8 8$ </td><td>0.2997 ± 0.0008</td><td> $0 . 3 0 3 0 \pm 0 . 0 0 0 4$ </td><td> $5 . 4 5 6 9 \pm 0 . 0 0 4 5$ </td><td> $\underline { { 1 . 2 1 7 4 \pm 0 . 0 1 1 6 } }$ </td><td> $3 . 2 7 7 5 \pm 0 . 0 0 9 5$ </td></tr></table>

Table 8: Quantitative comparison of fine-tuning SD3.5-M using different methods (RFT, FlowDPO, and FlowCPO) under the out-of-domain offline training setting. The fine-tuning data come from the open-source Open Image Preferences v1 Results dataset and were generated by models other than the SD3.5-M reference policy. We evaluate the models with several Classifier-Free Guidance (CFG) scales. Fine-tuned SD3.5-M variants are reported as mean ± standard deviation over 5 runs; pretrained baselines are listed as single evaluations. Within the SD3.5-M fine-tuning group, the best results are highlighted in bold, and the second best are underlined.
<table><tr><td>Model</td><td>CFG</td><td>GenEval</td><td>OCR</td><td>PickScore</td><td>ClipScore</td><td>HPSv2.1</td><td>Aesthetic</td><td>ImgRwd</td><td>UniRwd</td></tr><tr><td>Pretrained Model Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SD-XL</td><td></td><td>0.55</td><td>0.14</td><td>22.42</td><td>0.287</td><td>0.280</td><td>5.60</td><td>0.76</td><td>2.93</td></tr><tr><td>SD3.5-L</td><td></td><td>0.71</td><td>0.68</td><td>22.91</td><td>0.289</td><td>0.288</td><td>5.50</td><td>0.96</td><td>3.25</td></tr><tr><td>FLUX.1-Dev</td><td>一</td><td>0.66</td><td>0.59</td><td>22.84</td><td>0.295</td><td>0.274</td><td>5.71</td><td>0.96</td><td>3.27</td></tr><tr><td>SD3.5-M Fine-Tuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1.0</td><td>0.24</td><td>0.12</td><td>20.51</td><td>0.237</td><td>0.204</td><td>5.13</td><td>-0.58</td><td>2.02</td></tr><tr><td>Base Model</td><td>3.0</td><td>0.59</td><td>0.47</td><td>22.28</td><td>0.287</td><td>0.284</td><td>5.38</td><td>0.71</td><td>2.96</td></tr><tr><td></td><td>4.5</td><td>0.63</td><td>0.59</td><td>22.34</td><td>0.285</td><td>0.279</td><td>5.36</td><td>0.85</td><td>3.03</td></tr><tr><td></td><td>1.0</td><td>0.4398 ± 0.0086</td><td>0.1723 ± 0.0061</td><td>21.5913 ± 0.0194</td><td>0.2678 ± 0.0016</td><td>0.2611 ± 0.0010</td><td>5.4923 ± 0.0129</td><td>0.3110 ± 0.0216</td><td>2.5551 ± 0.0138</td></tr><tr><td>+ RFT [49, 5]</td><td>3.0</td><td>0.6650 ± 0.0076</td><td>0.5310 ± 0.0078</td><td>22.6427 ± 0.0108</td><td>0.2974 ± 0.0008</td><td>0.3032 ± 0.0002</td><td>5.4780 ± 0.0082</td><td>1.0514 ± 0.0134</td><td>3.1766 ± 0.0202</td></tr><tr><td></td><td>4.5</td><td>0.6845 ± 0.0078</td><td>0.5968 ± 0.0059</td><td>22.6854 ± 0.0063</td><td>0.2991 ± 0.0007</td><td>0.3074 ± 0.0003</td><td>5.4733 ± 0.0074</td><td>1.1228 ± 0.0133</td><td>3.2299 ± 0.0084</td></tr><tr><td></td><td></td><td>0.2290 ± 0.0084</td><td>0.1158 ± 0.0054</td><td>20.8273 ± 0.0155</td><td>0.2425 ± 0.0011</td><td>0.2264 ± 0.0006</td><td>5.7158 ± 0.0147</td><td>−0.3851 ± 0.0285</td><td></td></tr><tr><td>+ FlowDPO [33]</td><td>1.0</td><td>0.6065 ± 0.0141</td><td>0.4789 ± 0.0053</td><td>22.5399 ± 0.0212</td><td>0.2919 ± 0.0007</td><td>0.2917 ± 0.0008</td><td>5.5203 ± 0.0032</td><td>0.9063 ± 0.0109</td><td>2.2693 ± 0.0198</td></tr><tr><td></td><td>3.0 4.5</td><td>0.6513 ± 0.0132</td><td>0.5406 ± 0.0071</td><td>22.6201 ± 0.0165</td><td>0.2946 ± 0.0003</td><td>0.2993 ± 0.0003</td><td>5.4927 ± 0.0108</td><td>0.9934 ± 0.0164</td><td>3.0972 ± 0.0170 3.1694 ± 0.0117</td></tr><tr><td></td><td></td><td>0.4670 ± 0.0110</td><td>0.2351 ± 0.0027</td><td>21.8088 ± 0.0198</td><td>0.2762 ± 0.0011</td><td></td><td>5.3546 ± 0.0107</td><td></td><td></td></tr><tr><td>+ FlowCPO (β = 1, Ours)</td><td>1.0 3.0</td><td>0.6958 ± 0.0050</td><td>0.5561 ± 0.0085</td><td>22.4609 ± 0.0188</td><td>0.2966 ± 0.0003</td><td>0.2704 ± 0.0003 0.2952 ± 0.0004</td><td>5.4140 ± 0.0029</td><td>0.5230 ± 0.0140 1.0166 ± 0.0075</td><td>2.7723 ± 0.0152 3.2263 ± 0.0100</td></tr><tr><td></td><td>4.5</td><td>0.6995 ± 0.0060</td><td>0.5912 ± 0.0069</td><td>22.3570 ± 0.0111</td><td>0.2942 ± 0.0007</td><td>0.2942 ± 0.0002</td><td>5.3810 ± 0.0056</td><td>1.0159 ± 0.0089</td><td>3.2173 ± 0.0184</td></tr><tr><td></td><td>1.0</td><td>0.5706 ± 0.0124</td><td>0.2360 ± 0.0047</td><td></td><td>0.2813 ± 0.0006</td><td></td><td>5.3451 ± 0.0106</td><td></td><td></td></tr><tr><td> $+ \mathrm { F l o w C P O } \left( \beta = 0 . 5 , 0 \mathrm { u r s } \right)$ </td><td>3.0</td><td>0.6859 ± 0.0099</td><td>0.4919 ± 0.0071</td><td>22.0231 ± 0.0146 22.3229 ± 0.0064</td><td>0.2959 ± 0.0010</td><td>0.2752 ± 0.0005</td><td>5.3718 ± 0.0074</td><td>0.6294 ± 0.0064</td><td>2.8523 ± 0.0200</td></tr><tr><td></td><td>4.5</td><td>0.6815 ± 0.0040</td><td>0.4912 ± 0.0054</td><td>22.1181 ± 0.0128</td><td>0.2939 ± 0.0009</td><td>0.2882 ± 0.0003 0.2842 ± 0.0005</td><td>5.3073 ± 0.0069</td><td>0.9967 ± 0.0062 0.9452 ± 0.0108</td><td>3.1464 ± 0.0089 3.1068 ± 0.0117</td></tr></table>

## D.3 Additional Analyses Beyond Main Results

## D.3.1 Understanding the Role of Negative Regularization

Tab. 4 shows a clear pattern within the GenEval-only preference training setup: how negative samples enter the objective matters at least as much as whether they enter at all. Positive-only regularization provides a stable baseline and remains competitive across all three metric groups, especially under stronger CFG. Repulsive negative regularization is less reliable. Mild repulsion (λ = −0.1) stays close to positive-only training, but stronger repulsion (λ = −1) hurts most metrics, and overly aggressive repulsion (λ = −10) leads to numerical instability and collapsed training. This is consistent with the view that explicitly pushing the model away from dispreferred samples can over-amplify contrastive signals and destabilize optimization.

At moderate weights, attractive negative regularization gives the best trade-off in this ablation between alignment strength and training stability. The configuration with β = 0.5 and λ = 1 achieves the highest GenEval score (0.84), whereas repulsion with λ = −1 degrades all reported metrics. The configuration with $\beta = 1$ and λ = 0.1 also attains the best OCR (0.59), ImgReward (1.14), and UniReward (3.23). Because all models in this table are trained with GenEval-only supervision, these gains on typography and general-preference metrics should be interpreted as cross-metric transfer rather than direct optimization. Both signs diverge at magnitude 10, so the evidence supports moderate attractive matching rather than a blanket stability claim.

![](images/c414cf497537646a7f4b67b21282d56c2ebdd2728aceb3f4fa116f04e84a4070.jpg)  
Figure 3: Qualitative comparison on the GenEval benchmark. All samples are generated by models trained with in-domain preference data.

![](images/c3252758291dcd57d149ceb26594a962f0533f1dd8954cfc86a5dfc112ed3b46.jpg)  
Figure 4: Qualitative comparison on the OCR benchmark. All samples are generated by models trained with in-domain preference data.

## D.3.2 Hyperparameter Sensitivity and Ablation

To study the hyperparameter sensitivity of the proposed objective function (equation 13), we conduct ablations on the GenEval benchmark and track performance across training steps.

Flow Interpolation Coefficient (β). β controls how far the target flows move away from the reference prior. As shown in Fig. 9a, moderate values $( \beta \in [ 0 . 5 , 1 . 0 ] )$ perform best in this sweep on GenEval. In particular, $\beta = 0 . 5$ achieves the highest peak score.

![](images/b2cb19ddf2ec52bfbc445b64144714aae19b4a55b9c2c975e5a0c174d4f4dd0b.jpg)  
Figure 5: Qualitative comparison on the DrawBench benchmark. All samples are generated by models trained with in-domain preference data.

![](images/fe06224245e7df73cc316f353ee327fb383deb069c43de6bc4afc989f024b8f3.jpg)  
Figure 6: Qualitative comparison on the GenEval benchmark. All samples are generated by models trained with out-of-domain preference data.

Negative Regularization Weight (λ). Balancing positive alignment and negative regularization is important. Fig. 9b shows that a balanced choice (λ = 1.0) yields the most robust long-term performance among the tested values on GenEval.

Reference Prior Stability (η). FlowCPO relies on a stable reference prior inside the flow-matching objective. Fig. 9c shows that updating the reference model $v _ { o l d }$ with a high-EMA decay rate $( \eta = 0 . 9 9 )$ yields the most stable behavior in this sweep.

![](images/dc884e1b89f8a56c26919fd6bbf14488c249a08a652d83324596dc43399cb358.jpg)  
Figure 7: Qualitative comparison on the OCR benchmark. All samples are generated by models trained with out-of-domain preference data.

![](images/480297fc7738e192b45923321b0d9add3246b6929d4bd0931bf2530bbe899509.jpg)  
Figure 8: Qualitative comparison on the DrawBench benchmark. All samples are generated by models trained with out-of-domain preference data.

## D.4 Per-Prompt Diversity Evaluation

The primary empirical claim in RQ2 is that FLOWCPO provides stable offline preference optimization with a forward-KL objective in the evaluated regime, not that it universally produces more diverse samples than FlowDPO. To evaluate sample diversity directly, we compute the Vendi Score [11] separately over eight samples generated for each prompt and report the mean per-prompt score. We evaluate two classifier-free guidance (CFG) scales, 1.0 and 4.5. For each target column in Tab. 9, the quality score and Vendi Score are obtained from the corresponding specialist:

![](images/32fa62c42264e23a4daef5bb5e7f6c9edc5b62c7a376d770b311bd51bda49cf4.jpg)  
(a) Effect of β

![](images/288444746ffc0640742baeb83f1db4142f4be2dec70e0a9d10a1e1fd8ad8f13d.jpg)  
(b) Effect of λ

![](images/36b848d7f9a672a4702812d33967ee4b23b4061389e659b0338e30bf3c5fcac1.jpg)  
(c) Effect of η  
Figure 9: Hyperparameter sensitivity within the GenEval-only setup. We monitor the GenEval score during training to analyze the impact of (a) the flow interpolation coefficient β, (b) the negative loss weight λ, and (c) the EMA parameter η. The figure is intended to show relative stability trends for the tested values.

GenEval-only fine-tuning for GenEval, OCR-only fine-tuning for OCR, and multi-reward fine-tuning for PickScore, CLIPScore, HPSv2.1. The main entry in each cell is the quality score, while the parenthesized entry is the Vendi Score. Higher is better for both.

Table 9: Target quality and per-prompt diversity on SD3.5-M. Vendi Score is computed from eight samples per prompt and reported in parentheses. Each metric column evaluates the specialist fine-tuned for that target. Bold independently marks the highest quality score and the highest Vendi Score in each column.
<table><tr><td>Method</td><td>CFG</td><td>GenEval ↑ (Vendi ↑) OCR ↑ (Vendi ↑)</td><td>PickScore ↑ (Vendi ↑)</td></tr><tr><td rowspan="2">SD3.5-M</td><td>1.0 0.24 (2.1573)</td><td>0.12 (2.3073)</td><td>20.51 (2.2824)</td></tr><tr><td>4.5 0.63 (1.6540)</td><td>0.59 (1.7778)</td><td>22.34 (1.7742)</td></tr><tr><td rowspan="2">RFT</td><td>1.0 0.59 (1.8210)</td><td>0.35 (2.2417)</td><td>21.91 (1.9740)</td></tr><tr><td>4.5</td><td>0.75 (1.4547) 0.72 (1.4984)</td><td>22.57 (1.5544)</td></tr><tr><td rowspan="2">FlowDPO</td><td>1.0 0.59 (1.8926)</td><td>0.51 (2.5117)</td><td>20.72 (1.8671)</td></tr><tr><td>4.5</td><td>0.81 (1.5594) 0.75 (1.6310)</td><td>22.89 (1.5966)</td></tr><tr><td rowspan="2">FLOWCPO (β = 0.5)</td><td>1.0 0.76 (1.7385)</td><td>0.83 (2.0578)</td><td>22.48 (1.8936)</td></tr><tr><td>4.5</td><td>0.82 (1.4773) 0.86 (1.4955)</td><td>22.69 (1.5251)</td></tr></table>

Results. The results show a consistent quality–diversity trade-off: stronger CFG improves target quality while reducing per-prompt diversity. Across all comparisons in Tab. 9, increasing CFG from 1.0 to 4.5 raises the target quality score and lowers the corresponding Vendi Score. At CFG 1.0, FLOWCPO achieves the highest quality among the compared methods for all three targets (0.76 GenEval, 0.83 OCR, and 22.48 PickScore), while retaining higher diversity than its own CFG-4.5 outputs. In particular, lowering CFG from 4.5 to 1.0 increases the Vendi Score of FLOWCPO by 0.2612, 0.5623, and 0.3685 on the three targets, respectively, with corresponding quality decreases of 0.06, 0.03, and 0.21. However, FLOWCPO does not maximize raw Vendi Score: the pretrained model is most diverse in the GenEval and PickScore columns.

## E Broader Impacts and Release Considerations

Potential positive impacts. By turning preference alignment into a strictly offline optimization problem, our method can reduce the need for repeated online rollouts during alignment and make experimentation on continuous generative models more compute-efficient. Better alignment on structure-sensitive tasks such as compositional generation and text rendering can also improve the controllability and practical usefulness of text-to-image systems in benign applications such as design ideation, educational content creation, and accessibility-related graphics.

Potential negative impacts. The same improvement in preference alignment can increase the capability of image generators to produce more convincing synthetic content, including misleading text-in-image content or other deceptive media. In addition, the method can inherit biases from offline preference data and from the automatic reward signals used to construct winner/loser pairs, and the out-of-domain results in Sec. 5 together with the limitations discussed in Sec. 6 indicate that behavior may degrade when the offline data distribution differs from the reference model.

Release considerations. We do not release new model checkpoints, a new dataset, or a code package with this preprint; the paper describes the training objective and experimental protocol only. The experiments rely on existing pretrained models, public benchmarks, and a public OOD preference dataset under their original access conditions, and any future release of code or checkpoints should preserve the upstream licenses, terms of use, and usage restrictions of those assets.