# Beyond Random Couplings: Contrastive Noise Alignment in Generative Flows

Lennart Wittke<sup>1,</sup> <sup>2</sup> <sup>†</sup> Vinicius Azevedo<sup>2</sup>

<sup>1</sup>ETH Zürich <sup>2</sup>Disney Research | Studios

![](images/dbdb7426e6c6e877fa747fe97fcbd7e58ddaa299ce7e48ff5a427d6b88e3cc9f.jpg)  
Figure 1. Illustration of Contrastive Noise Alignment (CNA). We model the noise batch as interacting particles. A: Independent Couplings lack alignment with the data, resulting in unstructured couplings and highly crossed paths. B: CNA balances three forces on each particle $z _ { i } \colon ( 1 )$ semantic alignment attracts it to its target $x _ { i } ( \mathcal { L } _ { \mathrm { a l i g n } } ^ { + } )$ and repels it from unmatched targets $x _ { j } ~ ( \mathcal { L } _ { \mathrm { a l i g n } } ^ { - } ) ; ( 2 )$ angular repulsion between noise particles maximizes surface entropy $( { \mathcal { L } } _ { \mathrm { e n t r o p y } } ) ;$ and (3) radial gravity bounds the space $( \mathcal { L } _ { \mathrm { { n o r m } } } ) ^ { \bullet }$ . This yields structured couplings with significantly reduced path crossings while maintaining an approximately Gaussian distribution (i.e., a relaxed prior).

## Abstract

Diffusion and flow-matching models are typically trained by corrupting data through independently sampled Gaussian noise. While simple and scalable, this forward process induces arbitrary data-noise couplings,forcing the network to learn high-curvature transports between unrelated endpoints. Existing optimal-transport methods reduce this burden by reassigning fixed noise samples to data, but the source noise distribution itself remains passive. To address this, we introduce Contrastive Noise Alignment (CNA), a training-time method that creates dynamic, contrastive couplings by optimizing the noise representations directly. By modeling the noise batch as an interacting particle system, CNA employs a cross-modal InfoNCE objective to align noise particles with their paired data targets. To prevent spatial collapse, this alignment is regularized using an angular entropy term and a radial norm penalty. We show

theoretically that this equilibrium asymptotically preserves Gaussian structures, maintaining tractability during inference. Empirically, CNA improves the alignment between noise and data, reduces flow curvature, and provides better generation quality with fewer required sampling steps. For few-step, pixel-space generation (2-4 NFEs), CNA reduces FID by over 50% compared to standard rectified flow, and by at least 24% against Optimal Transport baselines.

## 1. Introduction

Modern diffusion and flow-matching models are commonly trained by coupling each data sample with an independently drawn Gaussian noise sample. This choice is attractive for obvious reasons: the Gaussian prior is analytically tractable, easy to sample, maximum-entropy under fixed first and second moments, and compatible with simple stochastic noising processes. Yet this convenience hides a geometric weakness. The coupling between data and noise is largely arbitrary: each image is paired with a randomly sampled endpoint, and the model must learn a denoising or velocity field whose regression target averages over many such unstructured pairings. At large scale this works remarkably well, but it also places the burden of discovering useful transport structure entirely on the network.

Recent work in flow matching makes this issue explicit. Minibatch optimal-transport couplings [6, 31, 36] reduce the complexity of the learned flow by replacing independent noise-data pairings with geometrically shorter assignments. Semidiscrete formulations [26] further improve scalability by exploiting the finite dataset structure, while normalizing-flow distillation [2] uses a pretrained invertible model to provide stronger coupling supervision. These methods suggest a common principle: generative training improves when the coupling between data and noise is not left entirely to chance. However, existing approaches either rely on simplified geometric costs [6, 31, 36], expensive transport solvers [26], or auxiliary models whose own training depends on prior coupling assumptions [2].

In this work, we ask a complementary question: instead of selecting a Gaussian endpoint independently from the data sample, can we construct a data-dependent endpoint that is still statistically indistinguishable from Gaussian noise? Rather than pushing the geometric burden entirely onto downstream regression or discrete assignments, we propose optimizing the noise representations directly at training time to construct data-dependent endpoints that remain statistically indistinguishable from standard Gaussian noise.

Motivated by contrastive learning [28], we introduce Contrastive Noise Alignment (CNA), which models the noise batch as an interacting particle system subject to a balance of attractive and repulsive forces. As we show theoretically, the combination of these forces asymptotically preserves standard multivariate Gaussian structure, maintaining prior tractability during inference.

We evaluate whether the proposed approach improves sample quality under matched compute and straightens generative trajectories. Our experiments compare CNA against standard independent coupling (I-CFM) and minibatch OT (OT-CFM) baselines across unconditional generation tasks on CIFAR-10 and ImageNet32. By shifting the burden of structure discovery into the prior itself, CNA provides a principled alternative to arbitrary random couplings.

In summary, our core contributions are:

• We propose Contrastive Noise Alignment (CNA), a training-time method that dynamically aligns noise representations with data targets to construct structured forward couplings while asymptotically preserving the standard Gaussian prior.

• We show that training flow models with these optimized noise particles leads to fewer sampling steps required to generate high-quality samples.

## 2. Related Work

Flow Matching and Optimal Transport. Continuous normalizing flows and diffusion models have driven recent advances in generative modeling [14, 22, 33]. Flow matching simplifies training by regressing vector fields along linear probability paths [23]. Under standard independent couplings, noise and data are paired randomly, causing marginal trajectories to cross and complicating velocity field regression. To mitigate this, methods like Minibatch OT [31, 36] and global alignments [18, 26] solve semi-discrete optimal transport problems to minimize transport cost. Additionally, Albergo and Vanden-Eijnden [1] introduce data-dependent couplings within stochastic interpolants. While these approaches find optimal permutations, they treat the underlying Gaussian prior as a rigid, static set. In contrast, our method actively reshapes the continuous spatial topology of the noise prior, moving beyond rigid matching to full distributional optimization.

Few-Step Sampling. Straightening generation trajectories inherently accelerates inference. Consequently, extensive literature explores model distillation [24, 34, 40], modified training objectives like Shortcut Models and MeanFlow [10, 11], and instance-aware discretization solvers [41]. Because these techniques modify the network’s regression objective or rely on pre-trained weights, they leave the foundational noise-data geometry unchanged. Our proposed prior optimization is fully orthogonal to these methods; by establishing a structurally aligned prior during the initial training phase, we produce inherently straighter paths that can further ease the downstream burden on few-step samplers.

Optimized Source Distributions. Recent work demonstrates that optimizing the initial source distribution can also yield straighter trajectories. Nayal et al. [27] propose MixFlow, which reduces path curvature by mixing conditioned and unconditional sources, while Kim et al. [17] learn a prior that adapts to specific conditioning variables. Unlike these methods, which introduce parameterized prior networks or mixture models, we rely entirely on an implicit, non-parametric contrastive update without adding architectural overhead.

Contrastive Objectives in Generative Modeling. The InfoNCE objective [28] is a staple of representation learning. Recently, Betser et al. [3] proved that the populationlevel InfoNCE objective asymptotically induces a Gaussian distribution when properly regularized. Within generative modeling, framing batch optimization as a particle-based objective has shown promise for maintaining global coverage [7]. While contrastive losses in continuous flows remain largely underexplored, Lee et al. [20] use them to improve text-to-image alignment, Kim et al. [15] apply them at inference time to enhance sample diversity, and Stoica et al. [35] separate conditional trajectories via contrastive velocity penalties. Our work is the first to leverage the asymptotic Gaussianity of contrastive learning to optimize the source distribution directly during flow matching training.

## 3. Preliminaries

## 3.1. Conditional Flow Matching and Stochastic Interpolants

Conditional Flow Matching [1, 22, 23, 36] provides a simulation-free approach to train continuous normalizing flows by constructing a time-dependent probability path between a tractable source noise distribution $\mu _ { 0 } ~ \in ~ \mathcal { P } ( \mathbb { R } ^ { d } )$ (typically $\mathcal { N } ( 0 , \bf { I } ) )$ and a complex target data distribution $\mu _ { 1 } \in \mathcal P ( \mathbb { R } ^ { d } )$ . Formally, let $\Pi ( \mu _ { 0 } , \mu _ { 1 } ) \subset \mathcal P ( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } )$ denote the set ofjoint distributions that have $\mu _ { 0 }$ and $\mu _ { 1 }$ as their marginals $( \mathrm { i . e . }$ ., the set of valid couplings). For any valid coupling $\pi \in \Pi$ , let $( x _ { 0 } , x _ { 1 } ) \sim \pi$ denote a paired sample, where $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { d }$ is drawn from the source marginal $\mu _ { 0 }$ and $x _ { 1 } \in \mathbb { R } ^ { d }$ is drawn from the target marginal $\mu _ { 1 }$ . Then, with an interpolant function $\phi _ { t } ( x _ { 0 } , x _ { 1 } ) \in \mathbb { R } ^ { d } \ , t \in [ 0 , 1 ] .$ subject to the boundary conditions $\phi _ { 0 } ( x _ { 0 } , x _ { 1 } ) = x _ { 0 }$ and $\phi _ { 1 } ( x _ { 0 } , x _ { 1 } ) = x _ { 1 }$ , the continuous probability path via the pushforward operation is defined as $\rho _ { t } ~ = ~ ( \phi _ { t } ) _ { \# } \pi , ~ \rho _ { t } ~ \in$ ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ . By construction, this ensures $\rho _ { 0 } = \mu _ { 0 }$ and $\rho _ { 1 } = \mu _ { 1 }$

The dynamics of this interpolant are governed by a probability flow ODE, driven by a time-varying marginal vector field $v _ { t }$ [1]:

$$
d x _ { t } = v _ { t } ( x _ { t } ) d t , \quad x _ { 0 } \sim \mu _ { 0 } .\tag{1}
$$

Directly learning $v _ { t }$ is intractable as it requires knowledge of the marginal density $\rho _ { t }$ . Instead, Flow Matching relies on constructing tractable conditional vector fields $u _ { t } ( x _ { t } \mid$ $x _ { 0 } , x _ { 1 } )$ that generate the conditional paths $\phi _ { t } ( x _ { 0 } , x _ { 1 } )$ . The marginal vector field is then recovered via marginalization: $v _ { t } ( x ) = \mathbb { E } _ { \pi } [ u _ { t } ( x _ { t } \mid x _ { 0 } , x _ { 1 } ) \mid x _ { t } = x ] .$

A widely used specific choice is the linear interpolant, defined as $x _ { t } : = \phi _ { t } ( x _ { 0 } , x _ { 1 } ) = ( 1 - t ) x _ { 0 } + t x _ { 1 } \ [ 2 2 ]$ . For this linear path, the conditional velocity term simplifies to a constant vector: $x _ { 1 } - x _ { 0 }$ . To learn this target vector field, a neural network $v _ { \theta } ( x _ { t } , t )$ is optimized to predict the flow by minimizing the conditional flow matching objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] } \left[ \| v _ { \theta } ( x _ { t } , t ) - ( x _ { 1 } - x _ { 0 } ) \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{2}
$$

Once $v _ { \theta }$ accurately models the true field $v _ { t } .$ , we can map pure noise into the target data distribution by evolving equation 1.

The choice of the coupling π significantly dictates the optimization dynamics. An independent coupling, $\pi ( x _ { 0 } , x _ { 1 } ) = \mu _ { 0 } ( x _ { 0 } ) \mu _ { 1 } ( x _ { 1 } )$ , pairs samples entirely at random. This stochastic construction averages out opposing directions, inducing highly curved marginal sampling trajectories.

OT-CFM. To mitigate this, Optimal Transport $( O T )$ couplings explicitly pair geometrically closer samples by minimizing a transport cost, typically the squared $L _ { 2 }$ distance:

$$
\pi _ { \mathrm { O T } } = \arg \operatorname* { m i n } _ { \pi \in \Pi ( \mu _ { 0 } , \mu _ { 1 } ) } \int \| x _ { 0 } - x _ { 1 } \| _ { 2 } ^ { 2 } d \pi ( x _ { 0 } , x _ { 1 } ) .\tag{3}
$$

To avoid intractable dataset-wide computations, practical implementations utilize Minibatch OT [31, 36]. By matching samples discretely per batch, this strategy empirically straightens the marginal vector fields.

## 3.2. Information Noise-Contrastive Estimation (InfoNCE)

Empirical InfoNCE Loss. We measure alignment using cosine similarity. For any vector $v \in \mathbb { R } ^ { d }$ , let $\hat { v } = v / \| v \| _ { 2 }$ denote its $L _ { 2 }$ -normalized counterpart, such that the cosine similarity between two vectors is given by their dot product $\langle \hat { z } , \hat { x } \rangle$ . By choosing this metric, the InfoNCE loss [4, $^ { 1 6 , }$ 28] evaluates alignment strictly based on angular distance, effectively operating on the unit hypersphere $\bar { \boldsymbol { S } } ^ { d - 1 }$

Let π denote the joint distribution of paired samples. Given a batch of $N$ pairs $\{ ( z _ { i } , x _ { i } ) \} _ { i = 1 } ^ { N }$ drawn i.i.d. from π, the empirical InfoNCE loss is formulated as

$$
\mathcal { L } _ { \mathrm { { I n f o N C E } } , \tau } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \langle \hat { z } _ { i } , \hat { x } _ { i } \rangle / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( \langle \hat { z } _ { i } , \hat { x } _ { j } \rangle / \tau ) }\tag{4}
$$

where a fixed temperature hyperparameter $\tau > 0$ regulates distribution sharpness.

Intuitively, the numerator maximizes the cosine similarity of the true positive pair $( z _ { i } , x _ { i } )$ . Simultaneously, the denominator computes a partition function over all available candidates $\{ x _ { j } \} _ { j = 1 } ^ { N }$ in the batch, where the $j \neq i$ instances serve as negative repulsors.

Population InfoNCE. As the batch size $N  \infty .$ , the empirical InfoNCE loss converges to a population-level functional at an $\mathcal { O } ( N ^ { - 1 / 2 } )$ rate [39]. In the symmetric case, where positive pairs $( z , x ) \sim \pi$ share identical marginals $\mu ,$ this limit elegantly decomposes the objective into two distinct geometric terms:

$$
\begin{array} { r l } {  { \operatorname* { l i m } _ { N \to \infty } ( \mathcal { L } _ { \mathrm { I n f o N C E } , \tau } - \log N ) } } \\ & { = - \frac { 1 } { \tau } \mathbb { E } _ { ( z , x ) \sim \pi } [ \langle \hat { z } , \hat { x } \rangle ] + \Phi _ { \tau } ( \mu ) } \end{array}\tag{5}
$$

where the second term is defined as:

$$
\Phi _ { \tau } ( \mu ) : = \mathbb { E } _ { z \sim \mu } \left[ \mathrm { l o g } \mathbb { E } _ { x \sim \mu } [ \mathrm { e x p } ( \langle \hat { z } , \hat { x } \rangle / \tau ) ] \right]\tag{6}
$$

The first term measures the alignment of positive pairs tightly coupled by π. The second term, $\Phi ( \mu )$ , acts as a uniformity potential that depends solely on the marginal distribution.

## 4. Proposed Method

## 4.1. Relaxed Prior Couplings

Standard continuous normalizing flows and optimal transport enforce strict marginal constraints $\pi ~ \in ~ \Pi ( \mu _ { 0 } , \mu _ { 1 } )$ ， where $\mu _ { 0 } = \mathcal { N } ( 0 , { \bf I } )$ is the prior and $\mu _ { 1 } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ is the data. This rigid boundary forces the transport map to resolve the entirety of the structural mismatch between an isotropic prior and clustered data, inevitably inducing tangled, high-curvature sampling trajectories.

To resolve this bottleneck, we extend the classical formulation into the regime of semi-unbalanced optimal transport [5]: rather than enforcing a strict boundary at $t = 0 .$ we allow the source distribution to be an adaptive measure $\tilde { \mu } _ { 0 } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ . We define relaxed prior couplings as joint measures $\pi \in \Pi ( \cdot , \mu _ { 1 } )$ , where the exact second marginal matches the data, but the first marginal, denoted ${ \tilde { \mu } } _ { 0 } ,$ remains a free variable. We argue that relaxing these rigid prior constraints is not a compromise but a necessity for fast sampling, significantly simplifying the transport map.

To systematically constrain the structural optimization of this adaptive source, we formulate a divergence-regularized transport objective. Assuming absolute continuity of the source with respect to the prior $( \tilde { \mu } _ { 0 } \ll \mu _ { 0 } )$ to ensure finite divergence, we define:

$$
\begin{array} { l } { \displaystyle \mathcal { W } _ { c , \mathrm { K L } } \big ( \mu _ { 0 } , \mu _ { 1 } \big ) = \operatorname* { i n f } _ { \boldsymbol { \pi } \in \Pi ( \cdot , \mu _ { 1 } ) } \Big [ - \frac { 1 } { \tau } \mathbb { E } _ { ( \boldsymbol { z } , \boldsymbol { x } ) \sim \boldsymbol { \pi } } \big [ \big < \hat { \boldsymbol { z } } , \hat { \boldsymbol { x } } \big > \big ] } \\ { \displaystyle + \beta \mathrm { K L } \big ( \tilde { \mu } _ { 0 } \mathrm { ~ } \| \mu _ { 0 } \big ) \Big ] } \end{array}\tag{7}
$$

where $\beta > 0$ controls the regularization strength.

This objective strategically shifts the modeling burden. The expected cosine similarity actively aligns the source marginal $\tilde { \mu } _ { 0 }$ to structurally mirror $\mu _ { 1 }$ , minimizing the required transport effort. Concurrently, the KL divergence term penalizes deviations from the reference Gaussian $\mu _ { 0 } .$ While $\beta  \infty$ recovers a hard-constrained OT problem with spherical cosine cost, a finite β yields an optimal intermediate source that perfectly balances semantic alignment with prior tractability.

## 4.2. Contrastive Noise Alignment

We propose Contrastive Noise Alignment (CNA) as a tractable, implicit empirical realization of relaxed prior couplings. Standard flow matching assumes a fixed prior $z \sim$ $\mathcal { N } ( 0 , \bf { I } )$ , forcing the velocity field to resolve severe structural mismatches via highly curved transport trajectories. CNA flips this paradigm: instead of treating a sampled noise batch as static vectors, we model it as a particle system. By actively optimizing the noise batch’s geometry to reflect the semantic topology of the target data, CNA effectively minimizes the relaxed objective (Equation 7).

We achieve this through an equilibrium of three geometric forces (Figure 1): (1) a cross-modal alignment force (transport cost) pulling noise toward assigned targets, (2) an angular entropy force maximizing directional diversity through repulsion, and (3) a radial gravitational force anchoring vectors to the origin. Together, the latter two explicitly tether the batch to the standard Gaussian prior, approximating the angular part of the KL penalty (App. C.1).

Initial Target Assignment. Directly optimizing the continuous joint measure π (Equation 7) is computationally prohibitive. To make this tractable, CNA employs an empirical relaxation by fixing a discrete initial coupling π<sub>init</sub> (drawn randomly or via minibatch OT, Equation 3) to pair each noise particle $z _ { i }$ with a target $x _ { i }$ . Crucially, $\pi _ { \mathrm { i n i t } }$ remains frozen during training. Rather than updating the coupling matrix, CNA optimizes the spatial coordinates of $z _ { i } .$ Under this strictly fixed assignment, the targets $x _ { i }$ act as stationary anchors, pulling the free-floating noise particles into a topologically aligned configuration.

Cross-Modal Alignment. To approximate the transport cost $c ( z , x )$ and actively sculpt the noise distribution, we align normalized noise particles $\hat { z } _ { i }$ with assigned targets $\hat { x } _ { i }$ via an InfoNCE objective. Given a batch of size $N .$ , the alignment loss is formulated as

$$
\mathcal { L } _ { \mathrm { a l i g n } , \tau } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \langle \hat { z } _ { i } , \hat { x } _ { i } \rangle / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( \langle \hat { z } _ { i } , \hat { x } _ { j } \rangle / \tau ) }\tag{8}
$$

where $\tau > 0$ is a temperature scaling parameter. By restricting the gradient updates to the noise particles, this optimization mechanism allows the noise vectors to self-organize into semantic clusters around the fixed data points, aligning the topology of the initial noise space with the underlying semantic structure of the data manifold. This is a fundamental advantage over standard Minibatch Optimal Transport [31, 36], which operates on a fixed, randomly sampled set of noise vectors and is thus restricted to point-to-point assignments.

The repulsive force in the denominator also serves as a key safeguard. By keeping the noise vectors repelled from the broader data manifold, it prevents them from collapsing directly into the target data. This leaves the local attraction just strong enough to capture the fine-grained details unique to the assigned target.

Noise-Noise Repulsion. To approximate the KL penalty and prevent spatial collapse, we apply a noise-noise repulsion force to the $L _ { 2 } .$ -normalized vectors $\hat { z } _ { i } \in \mathcal { S } ^ { d - 1 }$

$$
\mathcal { L } _ { \mathrm { e n t r o p y } , \gamma } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \sum _ { j \neq i } \exp \left( \frac { \langle \hat { z } _ { i } , \hat { z } _ { j } \rangle } { \gamma } \right) .\tag{9}
$$

By penalizing high cosine similarities, this objective treats vectors as mutually repelling charged particles, where $\gamma >$ 0 is a separate temperature parameter controlling the repulsive field.

Crucially, this is not merely a dispersion heuristic. $\mathbf { A } \mathbf { s }$ we formally demonstrate in Appendix $\mathrm { A . 1 }$ , minimizing this term is equivalent to maximizing the empirical angular entropy $\hat { H } ( \bar { Z } )$ , directly promoting the uniform distribution on the hypersphere in the asymptotic limit [39].

Gaussian Norm Regularizer. Because $\mathcal { L } _ { \mathrm { a l i g n } }$ and L<sub>entropy</sub> are scale-agnostic, optimizing them via discrete gradient steps provably (for GD) inflates vector magnitudes (see $\mathsf { A p - }$ pendix A.2). To counteract this centrifugal expansion and to preserve the radial constraint of the prior, we apply a stabilizing $L _ { 2 }$ penalty:

$$
\mathcal { L } _ { \mathrm { { n o r m } } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| z _ { i } \| _ { 2 } ^ { 2 } .\tag{10}
$$

Final Objective. The final objective of the CNA combines structural alignment $( \mathcal { L } _ { \mathrm { a l i g n } } )$ with decoupled angular and radial divergence penalties $( \mathcal { L } _ { \mathrm { e n t r o p y } }$ and ${ \mathcal { L } } _ { \mathrm { n o r m } } )$

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { a l i g n } , \tau } + \beta \mathcal { L } _ { \mathrm { e n t r o p y } , \gamma } + \lambda \mathcal { L } _ { \mathrm { n o r m } } .\tag{11}
$$

Iteratively minimizing this objective dynamically aligns the noise and target batches, yielding an optimized set of source particles $\{ \tilde { z } _ { i } \} _ { i = 1 } ^ { N }$ . These topologically aligned pairs $( \tilde { z } _ { i } , x _ { i } )$ replace the standard random samples $( x _ { 0 } , x _ { 1 } )$ in the Conditional Flow Matching objective (Equation 2). Importantly, our method only alters the training procedure, allowing inference to proceed exactly as usual. Implementation details are provided in Appendix B.

## 4.3. Theoretical Intuitions and Induced Gaussianity

The Core Intuition: Contrastive Noise Alignment (CNA) is grounded in the Maxwell-Poincaré spherical central limit theorem [8, 25, 30], which states that fixed-dimensional projections of a uniform distribution on a scaled hypersphere converge to a Gaussian as the dimension grows:

Lemma 1 (Maxwell-Poincaré; [8]). Let σ denote the uniform distribution on $S ^ { d - 1 }$ . As $d  \infty ,$ , for every fixed $k \geq 1$ , the k-dimensional marginal of $u \sim \sigma$ satisfies

$$
{ \sqrt { d } } u _ { k } \Rightarrow N ( 0 , \mathbf { I } _ { k } ) ,\tag{12}
$$

where $u _ { k }$ denotes the projection of u onto a fixed kdimensional subspace. The total variation distance between $\sqrt { d } u _ { k }$ and $\mathcal { N } ( 0 , \mathbf { I } _ { k } )$ converges at a rate of $\mathcal { O } ( d ^ { - 1 } ) ~ I ^ { 9 } J .$

By enforcing angular uniformity via ${ \mathcal { L } } _ { \mathrm { e n t r o p y } }$ and preventing magnitudes from exploding via ${ \mathcal { L } } _ { \mathrm { n o r m } }$ , our objective provides the necessary structural forces to drive the noise particles toward a standard normal distribution in high dimensions.

To formalize this mechanism, we map our setup to the population InfoNCE objective of Wang and Isola [39] $( \mathrm { i . e . }$ Equation 5) and a framework similar to that of Betser et al. [3]. Let $\mu _ { 0 } = \mathcal { N } ( 0 , { \bf I } )$ be the initial noise distribution, and ${ \tilde { \mu } } _ { 0 } = g _ { \# } \mu _ { 0 }$ be the pushforward distribution generated by our optimization loop $g .$ . Betser et al. [3] prove that optimizing a bounded contrastive objective forces $\tilde { \mu } _ { 0 }$ to minimize a regularized function, asymptotically inducing a Gaussian distribution.

Connection to Population InfoNCE. First, we observe that our empirical entropy loss $\mathcal { L } _ { \mathrm { e n t r o p y } , \gamma }$ asymptotically recovers the unimodal repulsive component of the population InfoNCE objective (Equation 5). Leveraging this equivalence, we can decouple the global objective into crossmodal and unimodal interactions, leading directly to the decomposition formalized in Theorem 1:

Theorem 1 (Asymptotic Decomposition). Let π denote the joint distribution of matched positive pairs $( z , x )$ , with $r e \mathrm { - }$ spective marginals $\tilde { \mu } _ { 0 }$ for the noise distribution and $\mu _ { 1 }$ for the target data. Let zˆ and xˆ denote their $L _ { 2 } .$ -normalized representations. For fixed parameters $\beta , \lambda > 0 ,$ as the batch size $N \to \infty ,$ our combined objective converges a.s. to:

$$
\begin{array} { l } { \displaystyle { \operatorname* { l i m } _ { N  \infty } \bigg ( \mathcal { L } _ { a l i g n , \tau } + \beta \mathcal { L } _ { e n t r o p y , \gamma } } } \\ { \displaystyle { \qquad + \lambda \mathcal { L } _ { n o r m } - ( 1 + \beta ) \log N \bigg ) = } } \\ { \displaystyle { - \frac { 1 } { \tau } \mathbb { E } _ { ( z , x ) \sim \pi } [ \langle \hat { z } , \hat { x } \rangle ] + \Phi _ { \tau } ( \tilde { \mu } _ { 0 } , \mu _ { 1 } ) } } \\ { \displaystyle { \qquad + \beta \Phi _ { \gamma } ( \tilde { \mu } _ { 0 } ) + \lambda \mathbb { E } _ { z \sim \tilde { \mu } _ { 0 } } [ \| z \| ^ { 2 } ] } } \end{array}\tag{13}
$$

where the unimodal repulsion $\Phi _ { \gamma } ( \tilde { \mu } _ { 0 } )$ and the cross-modal repulsion $\Phi _ { \tau } ( \tilde { \mu } _ { 0 } , \mu _ { 1 } )$ are defined as:

$$
\begin{array} { r l } & { \Phi _ { \gamma } ( \tilde { \mu } _ { 0 } ) = \mathbb { E } _ { z _ { i } \sim \tilde { \mu } _ { 0 } } \left[ \log \mathbb { E } _ { z _ { j } \sim \tilde { \mu } _ { 0 } } \left[ \exp ( \langle \hat { z } _ { i } , \hat { z } _ { j } \rangle / \gamma ) \right] \right] } \\ & { \Phi _ { \tau } ( \tilde { \mu } _ { 0 } , \mu _ { 1 } ) = \mathbb { E } _ { z \sim \tilde { \mu } _ { 0 } } \left[ \log \mathbb { E } _ { x \sim \mu _ { 1 } } \left[ \exp ( \langle \hat { z } , \hat { x } \rangle / \tau ) \right] \right] } \end{array}
$$

Proof. The full derivation is provided in Appendix A.3.

Induced Gaussianity. Let $\sigma$ denote the uniform distribution on $S ^ { d - 1 }$ . Crucially, Wang and Isola [39, Appendix $\mathbf { A } ]$ prove that the uniformity potential $\Phi _ { \gamma } ( \tilde { \mu } _ { 0 } )$ is uniquely minimized at the uniform distribution on the sphere (i.e., $\tilde { \mu } _ { 0 } = \sigma )$

<table><tr><td rowspan="3">Method</td><td colspan="8">FID ↓ (Euler NFEs)</td></tr><tr><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td></td><td>175.91</td><td>54.39</td><td>18.21</td><td>9.59</td><td>6.49</td><td>5.05</td><td>4.37</td></tr><tr><td>I-CFM OT-CFM [36]</td><td>346.76 229.98</td><td>91.75</td><td>30.00</td><td>14.20</td><td>9.08</td><td>6.58</td><td>5.12</td><td>4.36</td></tr><tr><td>CNA (ours, β = 2)</td><td>220.87</td><td>87.39</td><td>28.87</td><td>14.00</td><td>8.90</td><td>6.44</td><td>5.09</td><td>4.26</td></tr><tr><td>CNA + OT (ours, β = 2)</td><td>148.54</td><td>56.40</td><td>22.64</td><td>12.28</td><td>8.23</td><td>6.36</td><td>5.51</td><td>5.25</td></tr><tr><td> $\mathrm { C N A } + \mathrm { O T } \left( \mathrm { o u r s } , \beta = 5 \right)$ </td><td>179.84</td><td>69.59</td><td>25.24</td><td>12.93</td><td>8.27</td><td>5.88</td><td>4.62</td><td>4.06</td></tr></table>

(a) Unconditional generation results on CIFAR-10

![](images/d8156f7c7f73a102cbdbe03af13e85831057440200d38977cb19c9986c74d5ba.jpg)  
(b) FID across Euler NFEs.  
Figure 2. Quantitative results on CIFAR-10. (a) Table showing FID scores; (b) Plot visualizing performance versus sampling budget.

Corollary 1 (Induced Gaussianity as $\beta  \infty )$ . Under the conditions of Theorem 1 with initialization $\mu _ { 0 } = \mathcal { N } ( 0 , { \bf I } )$ and a scale-agnostic objective $( \lambda = 0 )$ , let $\tilde { \mu } _ { 0 }$ be the asymptotic global minimizer as $\beta \  \ \infty .$ Then, for any fixed $k \geq 1$ , the k-dimensional projections of the unnormalized representations $z \sim \tilde { \mu } _ { 0 }$ converge in distribution to $\mathcal { N } ( 0 , \mathbf { I } _ { k } )$ as $d \to \infty .$

The result follows from uniquely minimizing $\Phi _ { \gamma }$ at σ via Lemma 1, paired with the initial gaussian thin-shell concentration preserved under scale-invariance. See Appendix $\mathrm { A . 4 }$ for details. (Note: As discussed in Section 4 and $A p \mathrm { - }$ pendix A.2, empirical discrete optimization necessitates a norm regularizer to counteract the outward norm drift.)

Practical Considerations. While practical settings operate with finite dimensions and batch sizes, these theoretical limits still serve as valuable motivating approximations for training. Formally, the deviation of the expected contrastive loss from its population limit decays as $\overset { \mathcal { O } } { \left( M ^ { - 1 / 2 } \right) }$ in the number of negative samples M[39]. Meanwhile, the highdimensional projection error scales as $\mathcal { O } ( d ^ { - 1 } )$ [9]. When $\beta$ is sufficiently large, our experiments confirm that the optimized source distribution remains sufficiently close to a standard Gaussian prior.

## 5. Experiments

We evaluate CNA for unconditional image generation on CIFAR-10 and ImageNet32. We also perform an extensive ablation study to assess the contribution of each model component.

## 5.1. CIFAR-10 (Unconditional)

Experimental Setup. To validate our approach, we first test CNA on the standard CIFAR-10 benchmark, which contains 50, 000 training images of resolution 32 × 32. I-CFM, OT-CFM, and CNA training differ exclusively in the way noise-data pairs are sampled, all other aspects of FM training stay identical, using the setup from Tong et al. [36]. All reported baseline numbers are reproduced. During inference, we sample purely from the exact Gaussian prior, $x _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ . The full training details are provided in $\mathsf { A p - }$ pendix B.

Evaluation Metrics. We measure sample quality via Fréchet Inception Distance (FID) [13] and path straightness via flow curvature [21]. To test few-step generation, we report FID using a deterministic Euler solver across 1 to 128 number of function evaluations (NFEs).

Results. We present our quantitative results in Figure 2 and Table 1. Optimizing the noise prior via CNA consistently outperforms both I-CFM and OT-CFM across all generation budgets. At 2 sampling steps, CNA reduces FID from 175.91 to 87.39, a ∼50% reduction. This lead remains robust across few-step Euler budgets: CNA reaches 4-step and 8-step FIDs of 28.87 and 14.00, respectively $( \beta = 2 )$ Concurrently, CNA reduces flow curvature (Euler-128 κ drops ∼32% from 0.0479 to 0.0324), confirming that our aligned prior inherently straightens generation trajectories.

<table><tr><td></td><td>I-CFM</td><td>OT-CFM</td><td> $\overline { { \mathbf { C N A } \left( \beta = 2 \right) } }$ </td><td> $\overline { { \mathrm { C N A } { + } \mathrm { O T } \left( \beta = 2 \right) } }$ </td></tr><tr><td>Euler-128</td><td>0.0479</td><td>0.0326</td><td>0.0324</td><td>0.0251</td></tr><tr><td>Dopri5*</td><td>0.0531</td><td>0.0290</td><td>0.0280</td><td>0.0185</td></tr></table>

Table 1. Curvature κ (↓) for CIFAR-10. Note\*: Dopri5 κ deviates from the uniform-time definition.

The Initialization of CNA with OT couplings (CNA+OT) provides further gains across all evaluation budgets. Using $\beta = 2 ,$ , this combination reduces the 2-step FID to 56.40 (an ∼38% improvement over OT-CFM) and achieves the lowest curvature (κ = 0.0251 at Euler-128). At $\beta = 5 ,$ CNA+OT also improves the high-NFE limit, reaching an FID of 4.06 at Euler-128. This suggests that the optimized prior successfully retains the essential characteristics of a standard Gaussian prior if β is chosen high enough, while also providing a beneficial warm-start. By starting from an already efficient transport plan, the contrastive forces avoid large-scale structural reshuffling and can focus purely on fine-grained alignment. Qualitative samples reflect these metrics, producing much sharper images at low NFEs (Figure 3). Since CNA operates entirely during training, these benefits come at zero additional inference cost.

## 5.2. ImageNet32 (Unconditional)

To test the scalability of our approach on a significantly more complex and diverse data distribution, we extend our unconditional evaluation to the ImageNet32 dataset, which comprises approximately 1.28 million training images downsampled to a $3 2 \times 3 2$ resolution. As with the CIFAR-10 experiments, we maintain a fixed network architecture, regression objective, and training budget across all baselines, strictly isolating the impact of the noise coupling.

<table><tr><td rowspan="2">Method</td><td colspan="4">FID ↓ (Euler NFEs)</td></tr><tr><td>1</td><td>2</td><td>4</td><td>8 16</td></tr><tr><td>I-CFM</td><td>419.92</td><td>205.99</td><td>69.51</td><td>25.05 12.14</td></tr><tr><td>OT-CFM [36]</td><td>252.88</td><td>102.63</td><td>36.95 15.89</td><td>9.41</td></tr><tr><td>CNA (ours, β = 2)</td><td>250.87</td><td>105.67</td><td>39.93</td><td>17.85 10.89</td></tr><tr><td>CNA + OT (ours, β = 5) CNA + OT (ours, β = 30)</td><td>166.07 230.85</td><td>71.78 93.33</td><td>26.86 34.02</td><td>15.03 12.82 14.78 8.96</td></tr></table>

Table 2. Unconditional generation results on ImageNet32.

Results. Table 2 summarizes our quantitative evaluation on ImageNet32. CNA combined with OT initialization demonstrates substantial improvements in the highly constrained few-step regime. At $\beta \ = \ 5 ,$ our method drops the 1-step Euler FID from 252.88 (OT-CFM baseline) to 166.07—a massive ∼34% reduction. This robust performance extends through the 2-step and 4-step budgets, reducing FIDs to 71.78 (a ∼30% improvement) and 26.86, respectively. For higher step counts, increasing the entropy regularization $( \beta = 3 0 )$ secures the high-NFE limit, pushing the 16-step FID down to 8.96. As observed in previous experiments, these generation gains are structurally supported by a concurrent reduction in flow curvature (Table 3).

<table><tr><td></td><td>I-CFM</td><td>OT-CFM</td><td> $\overline { { \mathbf { C N A } \left( \beta = 2 \right) } }$ </td><td> $\overline { { \mathbf { C N A } + \mathbf { O T } \left( \beta = 1 0 \right) } }$ </td></tr><tr><td>Euler-128</td><td>0.0600</td><td>0.0416</td><td>0.0412</td><td>0.0381</td></tr><tr><td>Dopri5*</td><td>0.0629</td><td>0.0344</td><td>0.0326</td><td>0.0283</td></tr></table>

Table 3. Curvature κ (↓) for ImageNet32.

## 5.3. Analysis

Ablations. We perform a step-by-step ablation on CIFAR-10 to evaluate the contribution of each mechanism (Table 4). Applying only the contrastive alignment term pulls the noise distribution toward the data topology. This improves few-step generation, lowering the Euler-4 FID from 30.00 to 22.60 (at $\tau = 0 . 0 1 )$ . However, this unconstrained attraction induces spatial collapse (e.g., prior hole problem Hao and Shafto [12]), degrading adaptive solver quality (Dopri5 FID rises to 10.52).

<table><tr><td rowspan="2">Configuration</td><td colspan="3"> $\mathbf { F I D \downarrow }$ </td></tr><tr><td>Euler-4</td><td>Euler-10</td><td>Dopri5</td></tr><tr><td>OT-CFM (baseline) [36]</td><td>30.00</td><td>12.13</td><td>3.66</td></tr><tr><td>+Alignment  $( \tau = 0 . 1 )$ </td><td>25.32</td><td>15.81</td><td>15.84</td></tr><tr><td>+Alignment  $( \tau = 0 . 0 1 ) ^ { \mathrm { d e f } }$ </td><td>22.60</td><td>12.89</td><td>10.52</td></tr><tr><td>+Align + Ent  $( \tau , \gamma = 0 . 0 1 ) ^ { \mathrm { d e f } }$ </td><td>26.14</td><td>11.64</td><td>3.60</td></tr><tr><td> $\mathrm { C N A } + \mathrm { O T } ( \mathrm { A l i g n + E n t } + \mathcal { L } _ { \mathrm { n o r m } } )$ </td><td>25.24</td><td>11.02</td><td>3.56</td></tr></table>

Table 4. Step-by-step performance impact of Alignment, Entropy, and Norm regularization on CIFAR-10, starting from the OT-CFM baseline. Using β = 5. Default parameters are marked with <sup>def</sup>.

To address this, Entropy Repulsion acts as a uniformity potential against prior collapse, while Norm Regularization $( { \mathcal { L } } _ { \mathrm { n o r m } } )$ prevents the optimized vectors from escaping the $\mathcal { N } ( 0 , \bf { I } )$ shell. As the final table row indicates, combining all three components yields an optimal balance: The complete CNA formulation accelerates few-step generation (reducing Euler-4 and Euler-10 FIDs to 25.24 and 11.02) while restoring asymptotic quality (Dopri5 FID of 3.56).

![](images/c7132b5f0985a56edc1411ebfae5cfcd767496c652c50ec4e56d15b3d7f473d4.jpg)  
β

![](images/89798ef436bf7d10d94524e253c102ff79fb2071238865a0b5cfa01f21fecfe9.jpg)  
β  
Figure 4. Trajectory curvature and FID with regularization param eter β on CIFAR-10. Dotted lines denote the OT-CFM baseline.

Noise-Target Alignment. To verify structural alignment, we measure the average batch cosine similarity for the matched noise-image pairs (Table 5). While I-CFM yields zero correlation and OT-CFM provides an initial alignment (0.0379), our full CNA+OT (β = 5) method drives this further to 0.0597, an over 1.5× improvement over OT-CFM. A similar trend holds on ImageNet32, where CNA+OT (β = 10) increases alignment from 0.0491 to 0.0707. This confirms that CNA+OT systematically reduces directional displacement during transport.

<table><tr><td></td><td>I-CFM</td><td>OT-CFM</td><td>CNA (β = 2)</td><td>CNA+OT</td></tr><tr><td>CIFAR-10</td><td>0.000</td><td>0.0379</td><td>0.0453</td><td>0.0597</td></tr><tr><td>ImageNet32</td><td>0.000</td><td>0.0491</td><td>0.0561</td><td>0.0707</td></tr></table>

Table 5. Batch Correlation. Average cosine similarity θ between coupled pairs.

![](images/5b7353a4b1d2ad1c102994ef7c9588fce638fd190e220d653d1ede75d45cda3c.jpg)  
(a) CIFAR-10

![](images/bc9bb371f15e532e0f325256ee4358343c1fa822e4b209a05fa576ee350506be.jpg)  
(b) ImageNet32  
Figure 3. Qualitative comparison across different step counts. Compared to the baselines, our method (CNA + OT) maintains structura coherence even at very low NFEs (1 and 2 steps).

Effect of Regularization Weight $\beta .$ Figure 4 illustrates how the weight $\beta$ balances trajectory straightness and prior fidelity. We train different models with various values of $\beta$ on CIFAR-10. Keeping $\beta$ low allows the noise particles to align closely with the data. This creates exceptionally straight paths, resulting in peak few-step performance. The downside is that the prior drifts from a true Gaussian, which hurts high-step quality on solvers like Dopri5. Conversely, higher $\beta$ values reduce path straightness but benefit Dopri5 performance, even outperforming the baselines.

Post-Hoc Path Regulation. Interestingly, the flow matching network can be explicitly conditioned on the regularization parameter $\beta$ during training. This turns $\beta$ into a post-hoc inference parameter, allowing users to tune the trade-off between path straightness and prior adherence at generation time without retraining the model (for more details, see Table 6 and App. C.4).

<table><tr><td rowspan="2">Inference β</td><td colspan="4">FID↓</td></tr><tr><td>Euler-2</td><td>Euler-4</td><td>Euler-10</td><td>Dopri5</td></tr><tr><td>OT-CFM</td><td>91.75</td><td>30.00</td><td>12.13</td><td>3.66</td></tr><tr><td>β = 1.5</td><td>55.04</td><td>22.19</td><td>10.48</td><td>6.13</td></tr><tr><td>β= 3.0</td><td>62.11</td><td>23.83</td><td>10.54</td><td>4.25</td></tr><tr><td>β= 6.0</td><td>70.75</td><td>25.88</td><td>11.07</td><td>3.49</td></tr></table>

Table 6. Inference profile of the β-conditioned model on CIFAR-10. Lower $\beta$ values optimize for few-step generation.

Computational Overhead. While an inner optimization loop inevitably adds cost, the footprint of CNA remains manageable. Achieving effective alignment in just $T =$ 4 steps, it introduces a modest ${ \sim } 2 \%$ overhead on both CIFAR-10 and ImageNet32 (see Appendix C.3). CNA scales quadratically $( \mathcal { O } ( N ^ { 2 } ) )$ ) with batch size N, keeping it cheaper than the cubic $( \mathcal { O } ( N ^ { 3 } ) )$ exact Minibatch OT solvers [31, 36]. Consequently, in the combined CNA + OT variant, the OT initialization remains the primary runtime bottleneck. Additionally, by operating entirely online per minibatch, CNA bypasses offline pre-computation over the full dataset, which can reach hours at scale [26].

## 6. Conclusion

In this work, we introduced Contrastive Noise Alignment (CNA), a novel framework that optimizes the initial noise prior in flow matching models. CNA aligns source noise representations directly with the target data while approximately maintaining a global Gaussian distribution. In practice, this alignment significantly boosts generation quality when the sampling budget is limited. By offloading the task of discovering transport structure from the neural network to the prior itself, CNA provides a more direct, principled path toward learning geometrically aware forward processes.

Limitations & Ongoing Work. A key limitation of our method is the tension between aggressively aligning the noise prior and preserving its Gaussian properties. We are currently exploring new regularization methods to better constrain the adapted prior to the Gaussian manifold. Furthermore, while CNA optimizes the noise distribution, it is still initialized from random noise, meaning the starting point can be sub-optimal. A promising future direction is deterministic Gaussianization [19]: constructing the prior directly from the data to bypass random sampling entirely.

Beyond these theoretical questions, we are currently extending CNA to conditional and latent generation, as well as testing our contrastive loss within semantic feature spaces like DINOv2 [29]. Finally, because our approach is entirely orthogonal to the regression objective, we are excited to explore how it integrates with trajectory-straightening techniques, such as Shortcut Models and Mean Flow [10, 11].

## 7. Acknowledgments

We thank Pascal Chang for helpful discussions and feedback. We also acknowledge the support of ETH Zurich in providing access to the Euler cluster for this research.

## References

[1] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In International Conference on Learning Representations (ICLR), 2023. 2, 3

[2] David Berthelot, Tianrong Chen, Jiatao Gu, Marco Cuturi, Laurent Dinh, Bhavik Chandna, Michal Klein, Josh Susskind, and Shuangfei Zhai. The Coupling Within: Flow Matching via Distilled Normalizing Flows, 2026. arXiv:2603.09014 [cs]. 2

[3] Roy Betser, Eyal Gofer, Meir Yossef Levi, and Guy Gilboa. Infonce induces gaussian distribution. arXiv preprint arXiv:2602.24012, 2026. 2, 5, 3

[4] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020. 3

[5] Lenaic Chizat, Gabriel Peyré, Bernhard Schmitzer, and François-Xavier Vialard. Unbalanced optimal transport: Dynamic and kantorovich formulations. Journal of Functional Analysis, 274(11):3090–3123, 2018. 4

[6] Aram Davtyan, Leello Tadesse Dadi, Volkan Cevher, and Paolo Favaro. FASTER INFERENCE OF FLOW-BASED GENERATIVE MODELS VIA IMPROVED DATA-NOISE COUPLING. arXiv preprint arXiv:2603.15279, 2025. 2

[7] Mingyang Deng, He Li, Tianhong Li, Yilun Du, and Kaiming He. Generative Modeling via Drifting, 2026. arXiv:2602.04770 [cs]. 3

[8] Persi Diaconis and David Freedman. Asymptotics of graphical projection pursuit. The annals of statistics, pages 793– 815, 1984. 5

[9] Persi Diaconis and David Freedman. A dozen de finetti-style results in search of a theory. In Annales de l’IHP Probabilités et statistiques, pages 397–423, 1987. 5, 6

[10] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In International Conference on Learning Representations, pages 34668–34684, 2025. 2, 8

[11] Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. Advances in Neural Information Processing Systems, 38:75460–75482, 2026. 2, 8

[12] Xiaoran Hao and Patrick Shafto. Coupled variational autoencoder. arXiv preprint arXiv:2306.02565, 2023. 7

[13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6, 4

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 2

[15] Byungjun Kim, Soobin Um, and Jong Chul Ye. Diverse text-to-image generation via contrastive noise optimization. arXiv preprint arXiv:2510.03813, 2025. 3

[16] Daehee Kim, Youngjun Yoo, Seunghyun Park, Jinkyu Kim, and Jaekoo Lee. Selfreg: Self-supervised contrastive regularization for domain generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9619–9628, 2021. 3

[17] Junwan Kim, Jiho Park, Seonghu Jeon, and Seungryong Kim. Better source, better flow: Learning condition dependent source distribution for flow matching. arXiv preprint arXiv:2602.05951, 2026. 2

[18] Lingkai Kong, Molei Tao, Yang Liu, Bryan Wang, Jinmiao Fu, Chien-Chih Wang, and Huidong Liu. Alignflow: Im proving flow-based generative models with semi-discrete optimal transport. arXiv preprint arXiv:2510.15038, 2025. 2

[19] Valero Laparra, Gustavo Camps-Valls, and Jesús Malo. Iterative gaussianization: from ica to random rotations. IEEE transactions on neural networks, 22(4):537–549, 2011. 8

[20] Jaa-Yeon Lee, Byunghee Cha, Jeongsol Kim, and Jong Chul Ye. Aligning text to image in diffusion models is easier than you think. Advances in Neural Information Processing Sys tems, 38:157106–157136, 2026. 3

[21] Sangyun Lee, Beomsu Kim, and Jong Chul Ye. Minimizing trajectory curvature of ode-based generative models. In International Conference on Machine Learning, pages 18957– 18973. PMLR, 2023. 6, 4

[22] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Repre sentations (ICLR), 2023. 2, 3

[23] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Rep resentations (ICLR), 2023. 2, 3

[24] Xuesong Luo et al. Learning few-step diffusion models by trajectory distribution matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2

[25] James Clerk Maxwell. Ii. illustrations of the dynamical theory of gases. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 20(130):21–37, 1860. 5

[26] Alireza Mousavi-Hosseini, Stephen Y. Zhang, Michal Klein, and Marco Cuturi. Flow Matching with Semidiscrete Cou plings, 2026. arXiv:2509.25519. 2, 8

[27] Nazir Nayal, Christopher Wewer, and Jan Eric Lenssen. Mixflow: Mixed source distributions improve rectified flows. arXiv preprint arXiv:2604.09181, 2026. 2

[28] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 2, 3

[29] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 8

[30] Henri Poincaré. Calcul des probabilités. Gauthier-Villars, 1912. 5

[31] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky T. Q. Chen. Multisample Flow Matching: Straightening Flows with Minibatch Couplings, 2023. arXiv:2304.14772. 2, 3, 4, 8, 5

[32] Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. Advances in neural information processing systems, 29, 2016. 1

[33] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021. 2

[34] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning (ICML), 2023. 2

[35] George Stoica, Vivek Ramanujan, Xiang Fan, Ali Farhadi, Ranjay Krishna, and Judy Hoffman. Contrastive flow matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1185–1194, 2025. 3

[36] Alexander Tong, Nikolay Malkin, Guillaume Huguet, Jarrid Zhang, Jarrid Rector-Brooks, Kilian Fatras, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. In Transactions on Machine Learning Research (TMLR), 2024. 2, 3, 4, 6, 7, 8, 5

[37] Aad W Van der Vaart. Asymptotic statistics. Cambridge university press, 2000. 3

[38] Feng Wang, Xiang Xiang, Jian Cheng, and Alan Loddon Yuille. Normface: L2 hypersphere embedding for face verification. In Proceedings ofthe 25th ACM international conference on Multimedia, pages 1041–1049, 2017. 1

[39] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pages 9929–9939. PMLR, 2020. 3, 5, 6, 1, 2

[40] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step image translation with text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2

[41] Ming Yuan et al. Few-step diffusion sampling through instance-aware discretizations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 2

# Beyond Random Couplings: Contrastive Noise Alignment in Generative Flows Supplementary Material

We provide the following supplementary sections:

• Section A: Proofs and theoretical derivations.

• Section B: Implementation details, algorithm pseudocode, and hyperparameter settings.

• Section C: Further empirical analyses and theoretical discussions.

• Section D: Comparison of few-step generation performance.

• Section E: Additional qualitative results and generated visual samples.

• Section F: Statement on the use of large language models.

## A. Proofs

## A.1. Angular Entropy via von Mises-Fisher KDE

As demonstrated by Wang and Isola [39], minimizing the noise-noise repulsion term $\mathcal { L } _ { \mathrm { { e n t r o p y } } }$ is mathematically equivalent to maximizing the empirical differential entropy of the angular distribution, $H ( \hat { Z } )$

For an empirical batch of normalized vectors $\big \{ \hat { z } _ { 1 } , \dots , \hat { z } _ { N } \big \}$ , the empirical plug-in entropy is estimated via the sample average of the negative log-likelihood:

$$
\hat { H } ( \hat { Z } ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \hat { \mu } ( \hat { z } _ { i } )\tag{14}
$$

where $\hat { \mu } ( \hat { z } _ { i } )$ is the estimated angular density at point $\hat { z } _ { i }$ . To evaluate this density cleanly on the unit hypersphere $S ^ { d - 1 }$ , we employ a von Mises-Fisher (vMF) kernel density estimator (KDE) with a concentration parameter $\kappa = 1 / \gamma \colon$

$$
\hat { \mu } ( \hat { z } _ { i } ) = \frac { 1 } { C _ { \mathrm { v M F } } ( \gamma , d ) ( N - 1 ) } \sum _ { j \neq i } \exp \left( \frac { \langle \hat { z } _ { i } , \hat { z } _ { j } \rangle } { \gamma } \right)\tag{15}
$$

where $C _ { \mathrm { v M F } } ( \gamma , d )$ is the partition function (inverse normalization constant) for the vMF distribution in d dimensions. Substituting Equation 15 into Equation 14 yields:

$$
\begin{array} { r l } & { \hat { H } ( \hat { Z } ) = - \cfrac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } \log \left( \cfrac { 1 } { C _ { \mathrm { v M F } } ( \gamma , d ) ( N - 1 ) } \sum _ { j \neq i } \exp \left( \frac { \langle \hat { z } _ { i } , \hat { z } _ { j } \rangle } { \gamma } \right) \right) } \\ & { \quad \quad = - \left( \cfrac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } \log \sum _ { j \neq i } \exp \left( \frac { \langle \hat { z } _ { i } , \hat { z } _ { j } \rangle } { \gamma } \right) \right) + \log ( C _ { \mathrm { v M F } } ( \gamma , d ) ) + \log ( N - 1 ) } \end{array}\tag{16}
$$

We recognize the first expectation term on the right-hand side as the exact negative of our cross-sample particle repulsion loss, ${ \mathcal { L } } _ { \mathrm { e n t r o p y } }$ . Defining the constant $C = \log C _ { \mathrm { v M F } } ( \gamma , d ) + \log ( N - 1 )$ , we obtain the direct relation:

$$
{ \mathcal { L } } _ { \mathrm { e n t r o p y } } = - { \hat { H } } ( { \hat { Z } } ) + C\tag{17}
$$

Thus, minimizing the cross-sample particle interaction $\mathcal { L } _ { \mathrm { { e n t r o p y } } }$ is mathematically equivalent to maximizing the empirica alternative of the angular differential entropy $\hat { H } ( \hat { Z } )$ .

## A.2. Norm Growth during Discrete Noise Optimization

In this section, we analyze the idealized baseline case of discrete optimization under vanilla gradient descent $( \mathrm { G D } , \eta > 0 )$ with a scale-agnostic loss $\mathcal { L } ( \hat { z } )$ to demonstrate why unnormalized noise particles inherently expand outward. This centrifuga growth directly motivates our norm regularizer $\mathcal { L } _ { \mathrm { { n o r m } } }$ . The mathematical foundation for this phenomenon was established by Salimans and Kingma [32], Wang et al. [38]; here, we adapt to our specific noise optimization setting.

Let $z \in \mathbb { R } ^ { d }$ be an unnormalized noise particle, and let zˆ be its $L _ { 2 }$ -normalized counterpart on the unit hypersphere $S ^ { d - 1 }$ Suppose that we are minimizing a scale-agnostic loss $\mathcal { L } ( \hat { z } )$ . The gradient $g _ { t } = \nabla _ { z _ { t } } \mathcal { L }$ is computed via the multi-variable chain rule:

$$
g _ { t } = \left( \frac { \partial \hat { z } _ { t } } { \partial z _ { t } } \right) ^ { T } \nabla _ { \hat { z } _ { t } } \mathcal { L }\tag{18}
$$

Evaluating this derivative:

$$
\begin{array} { r l } & { \displaystyle \frac { \partial \hat { z } } { \partial z } = \frac { \partial } { \partial z } \left( z ( z ^ { T } z ) ^ { - \frac { 1 } { 2 } } \right) } \\ & { \quad = I ( z ^ { T } z ) ^ { - \frac { 1 } { 2 } } - z ( z ^ { T } z ) ^ { - \frac { 3 } { 2 } } z ^ { T } } \\ & { \quad = \displaystyle \frac { 1 } { \| z \| _ { 2 } } \left( I - \frac { z z ^ { T } } { \| z \| _ { 2 } ^ { 2 } } \right) } \\ & { \quad = \displaystyle \frac { 1 } { \| z \| _ { 2 } } \left( I - \hat { z } \hat { z } ^ { T } \right) } \end{array}\tag{19}
$$

Substituting back into the gradient formula:

$$
g _ { t } = \frac { 1 } { \Vert z _ { t } \Vert _ { 2 } } \left( I - \hat { z } _ { t } \hat { z } _ { t } ^ { T } \right) \nabla _ { \hat { z } _ { t } } \mathcal { L }\tag{20}
$$

This gradient g is strictly orthogonal to the current unnormalized noise vector z . We verify this identity by taking their inner $g _ { t }$ $z _ { t } .$ product:

$$
\begin{array} { r l } & { z _ { t } ^ { T } g _ { t } = z _ { t } ^ { T } \left[ \frac { 1 } { \| z _ { t } \| _ { 2 } } \left( I - \hat { z } _ { t } \hat { z } _ { t } ^ { T } \right) \nabla _ { \hat { z } _ { t } } \mathcal L \right] } \\ & { \quad \quad = \frac { 1 } { \| z _ { t } \| _ { 2 } } \left( z _ { t } ^ { T } - z _ { t } ^ { T } \frac { z _ { t } z _ { t } ^ { T } } { \| z _ { t } \| _ { 2 } ^ { 2 } } \right) \nabla _ { \hat { z } _ { t } } \mathcal L } \\ & { \quad \quad = \frac { 1 } { \| z _ { t } \| _ { 2 } } \left( z _ { t } ^ { T } - z _ { t } ^ { T } \right) \nabla _ { \hat { z } _ { t } } \mathcal L = 0 } \end{array}\tag{21}
$$

Because $z _ { t } ^ { T } g _ { t } = 0$ , every update step operates purely tangent to the hypersphere, seeking only to rotate the noise particles. Now, consider a discrete optimization update step with an optimization learning rate $\eta > 0 ;$

$$
z _ { t + 1 } = z _ { t } - \eta g _ { t }\tag{22}
$$

We compute the squared $L _ { 2 }$ norm of the updated particle $z _ { t + 1 }$ :

$$
\begin{array} { r l } & { \| z _ { t + 1 } \| _ { 2 } ^ { 2 } = ( z _ { t } - \eta g _ { t } ) ^ { T } ( z _ { t } - \eta g _ { t } ) } \\ & { \qquad = \| z _ { t } \| _ { 2 } ^ { 2 } - 2 \eta z _ { t } ^ { T } g _ { t } + \eta ^ { 2 } \| g _ { t } \| _ { 2 } ^ { 2 } } \end{array}\tag{23}
$$

Exploiting $( z _ { t } ^ { T } g _ { t } = 0 )$ , the cross-term vanishes, leaving:

$$
\| z _ { t + 1 } \| _ { 2 } ^ { 2 } = \| z _ { t } \| _ { 2 } ^ { 2 } + \eta ^ { 2 } \| g _ { t } \| _ { 2 } ^ { 2 } \geq \| z _ { t } \| _ { 2 } ^ { 2 }\tag{24}
$$

## A.3. Proof of Theorem 1

Proof. Let our combined objective be defined as $\mathcal { L } _ { \mathrm { t o t a l } } \triangleq \mathcal { L } _ { \mathrm { a l i g n } , \tau } + \beta \mathcal { L } _ { \mathrm { e n t r o p y } , \gamma } + \lambda \mathcal { L } _ { \mathrm { n o r m } }$ . We extend the proof of Theorem from [39].

By the Law of Large Numbers and the Continuous Mapping Theorem, as $N \to \infty ;$

$$
\begin{array} { r l } { \displaystyle \operatorname* { l i m } _ { N \to \infty } \left( \mathcal { L } _ { \mathrm { a l g n } , \tau } - \log N \right) = \displaystyle \operatorname* { l i m } _ { N \to \infty } \left[ - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \langle \hat { z } _ { i } , \hat { x } _ { i } \rangle / \tau + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \displaystyle \sum _ { j = 1 } ^ { N } \exp ( \langle \hat { z } _ { i } , \hat { x } _ { j } \rangle / \tau ) - \log N \right] } & { } \\ { = \displaystyle \operatorname* { l i m } _ { N \to \infty } \left[ - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \langle \hat { z } _ { i } , \hat { x } _ { i } \rangle / \tau + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \exp ( \langle \hat { z } _ { i } , \hat { x } _ { j } \rangle / \tau ) \right] } & { } \\ { = \displaystyle - \frac { 1 } { \tau } \mathbb { E } _ { ( z , x ) \sim \pi } [ \langle \hat { z } , \hat { x } \rangle ] + \mathbb { E } _ { z \sim \tilde { \mu } _ { 0 } } \left[ \log \mathbb { E } _ { x \sim \mu _ { 1 } } \left[ \exp ( \langle \hat { z } , \hat { x } \rangle / \tau ) \right] \right] } & { } \end{array}
$$

defining the second term as the cross-modal repulsion $\Phi _ { \tau } ( \tilde { \mu } _ { 0 } , \mu _ { 1 } )$

We apply the identical logic to our empirical entropy loss $\mathcal { L } _ { \mathrm { e n t r o p y } , \gamma } ,$ , which computes pairwise similarities strictly among N noise particles drawn from $\tilde { \mu } _ { 0 } { : }$

$$
\begin{array} { r l } { \displaystyle \operatorname* { l i m } _ { N \to \infty } \big ( \mathcal { L } _ { \mathrm { e n e p p . : ~ - l o g } } N \big ) = \displaystyle \operatorname* { l i m } _ { N \to \infty } \Bigg [ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \Bigg ( \displaystyle \sum _ { y \neq i } \exp ( \langle \xi _ { i } , \xi _ { i } \rangle / \eta ) \Bigg ) - \log N \Bigg ] } & { } \\ { = \displaystyle \operatorname* { l i m } _ { N \to \infty } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \Bigg ( \frac { 1 } { N } \sum _ { j \neq i } \exp ( \langle \xi _ { i } , \xi _ { j } \rangle / \eta ) \Bigg ) } & { } \\ { = \displaystyle \operatorname* { l i m } _ { N \to \infty } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \Bigg ( \frac { N - 1 } { N } \cdot \frac { 1 } { N - 1 } \sum _ { i \neq i } \exp ( \langle \xi _ { i } , \xi _ { j } \rangle / \eta ) \Bigg ) } & { } \\ { = \displaystyle \operatorname* { l i m } _ { N \to \infty } \log \Bigg ( \frac { N - 1 } { N } \Bigg ) + \displaystyle \operatorname* { l i m } _ { N \to \infty } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \log \Bigg ( \frac { 1 } { N - 1 } \sum _ { j \neq i } \exp ( \langle \xi _ { i } , \xi _ { j } \rangle / \eta ) \Bigg ) } & { } \\ { = \displaystyle \operatorname* { l i m } _ { N \to \infty } \log \Bigg ( \frac { N - 1 } { N } \Bigg ) + \displaystyle \operatorname* { s u p } _ { N \to \infty } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \log \Bigg ( \frac { 1 } { N } - 1 \sum _ { j \neq i } \exp ( \langle \xi _ { i } , \xi _ { j } \rangle / \eta ) \Bigg ) } & { } \\  = \mathrm { E . c . } \exp ( \log \mathrm { E . c . } / \eta _ { S } / \eta ) \Bigg [ \frac { \mathrm { E } } { \mathrm { E } } \ g _ { \mathrm { E } } / \eta ) \Bigg ] \equiv \Phi  \end{array}
$$

Summing these respective limits and scaling the unimodal repulsion explicitly by $\beta \colon$

$$
\operatorname* { l i m } _ { N \to \infty } \left( \mathcal { L } _ { \mathrm { t o t a l } } - ( 1 + \beta ) \log N \right) = - \frac { 1 } { \tau } \mathbb { E } _ { ( z , x ) \sim \pi } [ \langle \hat { z } , \hat { x } \rangle ] + \Phi _ { \tau } ( \tilde { \mu } _ { 0 } , \mu _ { 1 } ) + \beta \Phi _ { \gamma } ( \tilde { \mu } _ { 0 } ) + \lambda \mathbb { E } _ { z \sim \tilde { \mu } _ { 0 } } \left[ \| z \| ^ { 2 } \right]
$$

where we used the fact that lim $\begin{array} { r } { \mathfrak { r } _ { N \to \infty } \mathscr { L } _ { \mathrm { n o r m } } = \operatorname* { l i m } _ { N \to \infty } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| z _ { i } \| _ { 2 } ^ { 2 } = \mathbb { E } _ { z \sim \widetilde { \mu } _ { 0 } } \left[ \| z \| ^ { 2 } \right] } \end{array}$ . This proves Theorem 1. □

## A.4. Proof of Corollary 1

Proof. We adapt the proof from ([3], Appendix C). Let $z \in \mathbb { R } ^ { d }$ denote the unnormalized noise representation and write its polar decomposition $\mathrm { a s } ~ z = r u$ with $r = \| z \| _ { 2 } > 0$ and $u : = z / \| z \| _ { 2 } \in S ^ { d - 1 }$ . For any fixed $k \geq 1$ , let $P _ { k }$ be the corresponding orthogonal projector, setting $z _ { k } : = P _ { k } z$ and $u _ { k } : = P _ { k } u$

As $\beta \to \infty$ , the asymptotic objective reduces entirely to minimizing the uniformity potential $\Phi _ { \gamma } \left( \tilde { \mu } _ { 0 } \right) \left( \mathrm { i f } \lambda = 0 \right)$ . By Wang and Isola [39] (Appendix $\mathbf { A } ) , \Phi _ { \gamma }$ is uniquely minimized at the uniform law σ on $S ^ { d - 1 }$ , hence the angular component satisfies $u \sim \sigma$ . By the Maxwell-Poincaré spherical CLT (Lemma 1):

$$
{ \sqrt { d } } u _ { k } \Rightarrow { \mathcal { N } } ( 0 , \mathbf { I } _ { k } ) \quad ( d \to \infty ) .\tag{25}
$$

Because the scale-invariant objective evaluates strictly $L _ { 2 }$ normalized vectors, its continuous gradient flow acts purely tangentially to the hypersphere (see Appendix A.2). Under this idealized continuous optimization (and, $\mathrm { i . e . , } \lambda = 0 )$ , the radial distribution of the standard Gaussian initialization $\mu _ { 0 } ~ = ~ \mathcal { N } ( 0 , { \bf I } )$ is preserved. By high-dimensional gaussian thin-shell concentration, the normalized radius converges in probability:

$$
{ \frac { r } { \sqrt { d } } } \ { \stackrel { P } { \to } } \ 1 \quad ( d \to \infty ) .\tag{26}
$$

Since $\begin{array} { r } { z _ { k } = r u _ { k } = \left( \frac { r } { \sqrt { d } } \right) \left( \sqrt { d } u _ { k } \right) } \end{array}$ , combining the limits via Slutsky’s theorem [37] results in:

$$
z _ { k } \Rightarrow 1 \cdot \mathcal { N } ( 0 , \mathbf { I } _ { k } ) = \mathcal { N } ( 0 , \mathbf { I } _ { k } ) \quad ( d \to \infty ) .\tag{27}
$$

which concludes the proof of Corollary 1.

## B. Implementation Details

This section details the model architectures, training hyperparameters, and evaluation metrics used in our experiments.

## B.1. Evaluation Metrics

Fréchet Inception Distance (FID). FID evaluates sample quality by comparing the empirical statistics of real and generated images in the feature space of a pre-trained Inception-v3 network [13]. To match the baselines from other papers, we use the standard clean-fid implementation in legacy TensorFlow mode. Reference statistics are calculated on the training images and compared against 50,000 samples generated by the model.

Mean Path Curvature. We apply the trajectory curvature metric introduced by Lee et al. [21]. Let $v _ { t } ( x _ { t } )$ denote the predicted network velocity along the path from initial noise $x _ { 0 }$ to the generated data x . The mean path curvature κ is defined as:

$$
\kappa = \mathbb { E } _ { t , x _ { 0 } } \left[ \Vert ( x _ { 1 } - x _ { 0 } ) - v _ { t } ( x _ { t } ) \Vert _ { 2 } ^ { 2 } \right]\tag{28}
$$

where $t \sim U ( 0 , 1 )$ and $x _ { 0 } \sim \mu _ { 0 } = \mathcal { N } ( 0 , { \bf I } )$ . Curvature is not comparable across solvers and timestep schedules. However, within a fixed setup, lower values indicate straighter generative paths.

## B.2. Pseudocode

The detailed algorithm for our noise optimization method is provided in Algorithm 1.

```latex
Algorithm 1 Training Flow Matching with Contrastive Noise Alignment (CNA)
Inputs: Flow model $v _ { \theta }$ , batch size B, initial coupling $\pi _ { \mathrm { i n i t } } \in \Pi ( \mu _ { 0 } , \mu _ { 1 } )$ , CNA learning rate η, CNA optimization steps T,
temperatures τ, γ, loss weights $\beta , \lambda$
Outputs: Trained network parameters θ
1: while network θ has not converged do
2: Sample pairs $\{ ( z _ { i } , x _ { i } ) \} _ { i = 1 } ^ { B } \sim \pi _ { \mathrm { i n i t } }$
3: # Contrastive Noise Alignment (CNA)
4: $\hat { X }  \{ x _ { i } / \| x _ { i } \| _ { 2 } \} _ { i = 1 } ^ { B }$
5: for $t = 1$ to T do
6: $\hat { Z } \gets \{ z _ { i } / \| z _ { i } \| _ { 2 } \} _ { i = 1 } ^ { B }$
7: $\mathcal { L } _ { \mathrm { a l i g n } } : = \mathrm { I n f o N C E } ( \hat { Z } , \hat { X } , \tau )$
8: $\begin{array} { r } { \mathcal { L } _ { \mathrm { e n t r o p y } } : = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \left( \sum _ { j \neq i } \exp \left( \frac { \hat { z } _ { i } \cdot \hat { z } _ { j } } { \gamma } \right) \right) } \end{array}$
9: $\begin{array} { r } { \mathcal { L } _ { \mathrm { n o r m } } : = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } | | z _ { i } | | _ { 2 } ^ { 2 } } \end{array}$
10: $\mathcal { L } _ { \mathrm { C N A } } : = \bar { \mathcal { L } } _ { \mathrm { a l i g n } } + \beta \mathcal { L } _ { \mathrm { e n t r o p y } } + \lambda \mathcal { L } _ { \mathrm { n o r m } }$
11: $Z \gets Z - \eta \cdot \nabla _ { Z } \mathcal { L } _ { \mathrm { C N A } }$ {▷ Update noise states}
12: end for
13: Let $Z ^ { * } = \{ z _ { i } ^ { * } \} _ { i = 1 } ^ { B }  Z$ be the optimized coupled noise batch
14: $\theta  \mathrm { F M S t e p } ( \theta , ( z _ { i } ^ { * } ) _ { i = 1 } ^ { B } , ( x _ { i } ) _ { i = 1 } ^ { B } )$ {▷ Sample t, compute vectorfield loss, update θ}
15: end while
16: return θ
```

## B.3. Flow Model Training

For each dataset, the specific U-Net architectural choices and hyperparameters are detailed in Table 7.

CIFAR-10. To ensure a fair comparison with prior work, we adopt the exact ADM architecture and hyperparameters specified by Tong et al. [36]. For this dataset, all models are trained using a single NVIDIA RTX 4090 GPU.

ImageNet. We follow the U-Net configuration and hyperparameters described by Pooladian et al. [31]. These models are trained on a single node with 8× NVIDIA RTX 4090 GPUs.

For all models, we also apply an Exponential Moving Average (EMA) with a decay factor of 0.9999 to the weights.

## B.4. Hyperparameter Settings

For our main experiments, we evaluate our methods on CIFAR-10 and ImageNet (32 × 32). The complete configuration of optimization parameters, temperatures, and loss weights is detailed in Table 8.

For both datasets, the number of optimization steps is set to $N _ { \mathrm { o p t } } = 4 \mathrm { u s i n g } A d a m$ optimizer with default settings and a learning rate of $\eta = 0 . 0 2$ . The temperature parameters controlling feature scaling and field repulsion are fixed at $\tau = 0 . 0 1$

<table><tr><td colspan="2">CIFAR-10</td><td>ImageNet (32x32)</td></tr><tr><td>Channels</td><td>128</td><td>256</td></tr><tr><td>Depth</td><td>2</td><td>3</td></tr><tr><td>Channels multipliers</td><td>1,2,2,2</td><td>1,2,2,2</td></tr><tr><td>Heads</td><td>4</td><td>4</td></tr><tr><td>Heads channels</td><td>64</td><td>64</td></tr><tr><td>Attention resolution</td><td>16</td><td>4</td></tr><tr><td>Dropout</td><td>0.1</td><td>0.0</td></tr><tr><td>Batch size / GPU</td><td>128</td><td>128</td></tr><tr><td>Effective batch size</td><td>128</td><td>1024</td></tr><tr><td>GPUs</td><td>1</td><td>8</td></tr><tr><td>Iterations</td><td>390k</td><td>400k</td></tr><tr><td>Learning rate</td><td>0.0002</td><td>0.0001</td></tr><tr><td>Learning rate scheduler</td><td>Constant</td><td>Polynomial Decay</td></tr><tr><td>Warmup steps</td><td>5k</td><td>20k</td></tr><tr><td>Model parameters</td><td>36M</td><td>189M</td></tr></table>

Table 7. ADM network architecture and hyperparameters used for flow model training.

<table><tr><td>(a) Shared Parameters</td><td colspan="4">(b) Tuned Parameters</td></tr><tr><td></td><td colspan="2">CNA</td><td colspan="2">CNA + OT</td></tr><tr><td>Optimization steps  $N _ { \mathrm { o p t } }$  4 Learning rate η 0.02</td><td>β</td><td>λ</td><td>β</td><td>λ</td></tr><tr><td></td><td>1-5</td><td>0.001</td><td>1-10</td><td>0.001</td></tr><tr><td></td><td>1-5</td><td>0.001</td><td>1-50</td><td>{0.001, 0.005}</td></tr></table>

Table 8. Hyperparameter configurations for CNA and $\mathrm { C N A } + \mathrm { O T } .$ Shared optimization settings are applied across both strategies and datasets.

and $\gamma = 0 . 0 1$ , respectively. Furthermore, the norm regularization weight λ is tuned such that the average $L _ { 2 }$ norm of the optimized batch is preserved. We evaluate two initialization strategies: standard independent coupling (CNA) and pre structured initialization via minibatch OT (CNA + OT) [31, 36].

## C. Further Analyses And Discussion

## C.1. Connection to Kullback-Leibler Minimization

To understand what our regularizer actually targets, we can connect it to the Kullback-Leibler (KL) divergence between our optimized prior $q ( z )$ and the target standard Gaussian $\mathcal { N } ( 0 , \bf { I } )$

$$
D _ { \mathrm { K L } } ( q \parallel \mathcal { N } ( 0 , { \bf I } ) ) = - h ( q ) + \frac { 1 } { 2 } \mathbb { E } _ { z \sim q } \left[ \Vert z \Vert _ { 2 } ^ { 2 } \right] + \mathrm { c o n s t . }\tag{29}
$$

where $h ( q )$ is the joint differential entropy. Decomposing z into polar coordinates $z \ = \ r u$ (where $r \ = \ \| z \| _ { 2 }$ and $u \_ =$ $z / \| z \| _ { 2 } \in S ^ { d - 1 } )$ , this joint entropy expands as:

$$
h ( q ) = h ( u ) + h ( r \mid u ) + ( d - 1 ) \mathbb { E } [ \log r ]\tag{30}
$$

where $h ( u )$ is the angular entropy on the hypersphere and $h ( r \mid u )$ is the conditional radial entropy.

Comparing this decomposition to our regularizer $L _ { \mathrm { r e g } } = \beta \mathcal { L } _ { \mathrm { e n t r o p y } } + \lambda \mathcal { L } _ { \mathrm { n o r m } }$ , we see that while we optimize the angular entropy $h ( u )$ (via $\mathcal { L } _ { \mathrm { e n t r o p y } } )$ and penalize the second moment (via ${ \mathcal { L } } _ { \mathrm { n o r m } } )$ , our objective lacks the conditional radial entropy $h ( r \mid u )$

Thus, our method does not minimize the full joint KL divergence; instead, it is best understood as a proxy that approximates the angular part of the KL penalty, relying on ${ \mathcal { L } } _ { \mathrm { n o r m } }$ purely as an empirical scale stabilizer (see Appendix A.2).

## C.2. Empirical Verification of Gaussianity

To verify that CNA preserves the structure of the Gaussian prior during training, we investigate the optimized noise across two key geometric properties: radial concentration and spatial isotropy.

Radial concentration. We analyze the radial concentration of noise particles to verify whether the optimization preserves the $L _ { 2 }$ scale of the Gaussian prior. In high dimensions, a standard Gaussian vector $\boldsymbol { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } )$ concentrates on a thin spherical shell following a Chi distribution $\chi _ { d } .$ Figure 5 compares the norm density of the initial noise against optimized noise with and without the radial norm penalty ${ \mathcal { L } } _ { \mathrm { n o r m } }$ . Because cosine alignment $( \mathcal { L } _ { \mathrm { a l i g n } } )$ and angular entropy $( \mathcal { L } _ { \mathrm { e n t r o p y } } )$ operate on normalized vectors, optimizing without ${ \mathcal { L } } _ { \mathrm { n o r m } }$ causes norm expansion, shifting the density rightward. In contrast, adding our radial norm penalty $( \lambda = 0 . 0 0 1$ , for $\beta = 5 )$ counteracts this drift, keeping the spatial distribution virtually unchanged.

![](images/42e27988fef57b193259edd644822a9ee7d45757db8e8d6eb2573ba6ca43d75d.jpg)  
Figure 5. Radial concentration density of noise particles $( d = 3 0 7 2 )$ . When trained without the radial norm penalty $( \mathcal { L } _ { \mathrm { n o r m } } ) .$ , scale-agnostic contrastive alignment induces centrifugal expansion, shifting particle norms rightward. Including ${ \mathcal { L } } _ { \mathrm { { n o r m } } }$ successfully prevents this.

Preserving isotropic structure. We run Principal Component Analysis (PCA) on optimized noise batches to check whether CNA maintains isotropic variance across dimensions. Figure 6 shows the eigenvalue decay across regularization strengths (β). Without structural penalties $( \beta = 0 )$ , the noise over-adapts and leading dimensions dominate. Introducing entropy and norm penalties $( \beta = 1 , 5 )$ pulls the spectrum back to the baseline. At $\beta = 5$ , the decay matches standard random noise, confirming that repulsive forces prevent spatial collapse while keeping the Gaussian prior intact.

![](images/065f5900606590fe72920c79a939ddac1c471af73174b4c71206ec3393b715c2.jpg)  
Figure 6. Eigenvalue decay (log scale) of the source noise distributions across different regularization strengths.

## C.3. Computational Efficiency

We analyze the scalability of our method via training throughput as shown in Table 9. CNA introduces minimal overhead relative to independent couplings. Because the inner optimization converges in just a few steps, the actual time penalty is practically negligible, costing less than $\mathbf { a } \sim 2 \%$ drop in throughput. Even at larger batch sizes $( N = 1 0 2 4 )$ where the exact OT solver becomes a noticeable bottleneck, our contrastive updates scale effortlessly without adding any extra drag.

<table><tr><td>Dataset</td><td>I-CFM</td><td>CNA (ours)</td><td>OT-CFM</td><td>CNA + OT (ours)</td></tr><tr><td>CIFAR-10 (N = 128)</td><td>6.68</td><td>6.58</td><td>6.65</td><td>6.56</td></tr><tr><td>ImageNet32 (N = 1024)</td><td>1.77</td><td>1.75</td><td>1.31</td><td>1.30</td></tr></table>

Table 9. Training throughput comparison (Iterations / s, measured on Nvidia RTX 4090).

## C.4. Post-Hoc Path Regulation via β-Conditioning

We train an adaptive variant of our model where the entropy regularization weight $\beta$ is sampled uniformly from the interval [1, 6] during training. To handle this continuous conditioning, we compute a sinusoidal embedding of $\beta$ passed through a 2-layer $\mathbf { M L P } \psi ( \beta )$ , which is directly added to the standard time embedding ϕ(t). This explicitly conditions the learned vector field $v _ { \theta } ( x _ { t } , t , \beta )$ on the regularization parameter.

Intuitively, varying $\beta$ parameterizes a family of transport trajectories between the prior and the data. This allows the mode to learn multiple routing strategies, ranging from straight, highly regularized paths to complex, entropic ones.

At inference time, we sample directly from a standard Gaussian prior $x _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ and supply a fixed $\beta$ to guide the generation path. Table 6 presents the CIFAR-10 results. Tuning $\beta$ at inference shifts the performance profile across differen step budgets, allowing users to dynamically balance generation quality and path alignment post-hoc without retraining the model.

## C.5. Qualitative Analysis of Prior-Data Alignment

Figure 7 provides a visual demonstration of noise-data alignment. While standard Gaussian noise (middle row) shares nearzero spatial correlation with the target image, our aligned noise achieves a much higher cosine similarity with the target data while preserving its Gaussian characteristics.

![](images/0b5462d1e9fb90d7d20cbc3ee1fae9df10e218440a4f0ffa8d1ccad463754b3d.jpg)  
Figure 7. Visual comparison of original CIFAR-10 data (top), standard initial Gaussian noise (middle), and the corresponding optimized noise (bottom). The cosine similarity r between the original image and the noise is reported below each sample.

## D. Few-Step Sampling Quality

To illustrate how CNA performs in the few-step regime, Figure 8 compares qualitative results across different sampling budgets. As seen in the figure, CNA consistently produces higher quality samples than the baselines, especially when constrained to just 2 or 4 steps.

## E. Additional Qualitative Results

We provide additional qualitative examples that are generated using CNA. All images are produced with the Euler solver at 64 sampling steps. Results for CIFAR-10 and ImageNet32 are shown in Figures 9 and 10, respectively.

## F. Use of Large Language Models

We used Large Language Models (LLMs) solely for the refinement and polishing of this paper. The core research, including the methodological formulation, experimental design, and empirical analysis, remains the exclusive work of the authors.

![](images/e078ab409d6f50878fca5043ef21542a1029e12de8cdf55750c81248c450cd6e.jpg)  
(a) CIFAR-10

![](images/c5b36ab25ca03ced0f950ae614c60dcc43307ca8075e5f1dcaae789245e8d551.jpg)  
(b) ImageNet32  
Figure 8. Qualitative comparison across different step counts. Comparison of sampling trajectories for CIFAR-10 (left) and ImageNet32 (right). At 1 and 2 NFEs, our method maintains significantly better global structure than the baseline variants.

![](images/02205aa9bb0aed5de30f34247551f3316ffd4bb8c4f436d26446b3b257703de2.jpg)  
Figure 9. Uncurated 32×32 samples on CIFAR-10 (β = 5).

![](images/8ce63275510e05ae1850a3eb257808ff499e544e69d2268d9059271c879e862c.jpg)  
Figure 10. Uncurated 32×32 samples on ImageNet32 (β = 30)