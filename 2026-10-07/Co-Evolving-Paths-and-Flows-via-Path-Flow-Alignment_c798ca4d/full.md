# Co-Evolving Paths and Flows via Path–Flow Alignment

Zeyu Michael Li<sup>∗</sup> Duke University

William Xingxu Chen Duke University

Xiang Cheng Duke University

## Abstract

We study path–flow alignment as a unified training objective for flow matching. Instead of fixing the interpolation path and learning only the velocity field, we jointly train an endpoint-preserving path network and a flow network using the same alignment loss: the flow learns to match the path velocity, and the path learns to align its velocity to the current flow. Although every fixed learned path defines a valid flow-matching objective, the alignment loss alone is not a reliable criterion for path learning. We identify path overfitting, a failure mode in which the alignment loss decreases while sample quality worsens. We find that this failure is associated with low-entropy bottlenecks in the induced probability path, where the learned path routes samples through overly concentrated intermediate marginals. Motivated by this diagnosis, we introduce a stochastic path regularizer that hides part of the source information from the path network while preserving exact endpoints. The resulting regularization gives an explicit entropy floor for the stochastic training-path marginals and empirically suppresses the bottleneck in the learned sampler, making joint path–flow training effective. On ImageNet-256×256 with SiT backbones, our method consistently improves FID across model scales, extends to model-guidance training, and leaves the inference-time architecture and sampler unchanged. Code is available at https://github.com/lizeyu090312/ traj\_opt\_paper.

## 1 Introduction

Flow matching learns a generative ODE through a supervised regression problem: sample a source point and a data point, interpolate between them, and train a velocity field to match the tangent of that interpolation. This view underlies flow matching and stochastic interpolants [11, 2, 1], and scales to modern image backbones such as SiT [13]. In most large-scale image models this interpolation is fixed in advance, often as the linear path

$$
\gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) = ( 1 - t ) x _ { 0 } + t x _ { 1 } .
$$

The interpolating path is not a passive implementation detail. It determines both the states on which the model is trained and the vector targets it is asked to predict. Prior work has improved flow matching by changing endpoint couplings, straightening trajectories, using metric geometry, or learning curved paths [12, 19, 15, 8, 17]. We ask a complementary question:

What makes a path good, and can we learn itjointly with theflow?

A natural answer is to use the flow-matching objective itselfas a self-supervising signalfor both the path and theflow – a good path should make the flow-matching regression problem easier. Motivated by this, we propose path–flow alignment: we parameterize a path network $\gamma _ { \psi } .$ , and jointly train the path network and the flow network on the same path–flow alignment loss. We illustrate how learning the interpolation path changes the trajectories of the resulting flow in a toy example (Figure 1).

![](images/65293ab3c9ba895db251feda28ea4d38f91f2e91b87139bf385149f6c7ad4d16.jpg)

![](images/abe3479874bc5c38c4cffad6f8c622e0166fdc7b4cf9c851fa9d3a20080c44ac.jpg)  
Figure 1: Learned ODE trajectories in a toy mixture example. Top: standard flow matching, path–flow alignment without stochastic regularization $( \rho = 1 )$ , and with stochastic regularization $( \rho _ { t } = 1 - t )$ shown at three times. Bottom: standard and unregularized trajectories overlaid. All methods start from the same source samples. Appendix F.4 gives further details.

For any fixed learned path, we verify that this remains a valid flow-matching problem. Empirically, we observe that this approach quickly improves the flow-matching loss. However, the generation quality soon begins to degrade even as flow-matching loss continues to improve.

We call this path overfitting. In our sweeps, such overfitting is consistently accompanied by entropy collapse of intermediate marginals – intuitively, the path network $\gamma _ { \psi }$ is routing most of the flow through a series of low-entropy bottlenecks. This proves our first answer to be incomplete: a good path should not only reduce flow-matching loss, but do so while avoiding entropy collapse.

Motivated by this diagnosis, we introduce a simple regularization scheme that applies a Gaussian perturbation to the input of the path network. Theoretically, we show that, under the independent endpoint coupling used here, this regularization establishes an entropy floor on the stochastic training path marginals. Empirically, we show that this stochastic-regularized path–flow alignment effectively improves the generation quality on the challenging image generation tasks. We also show that the same path–flow alignment template extends beyond standard flow matching by applying it to model-guidance training [18]. Overall, our results demonstrate the promise of joint training of path and flow networks, assuming proper regularization.

## 1.1 Contributions

Our main contributions are as follows.

1. Learning interpolation paths for flow matching. We introduce path–flow alignment (Algorithm 1), a training-time method for learning endpoint-preserving neural paths jointly with a flow model. Section 3 defines the construction; Proposition 1 gives the correctness guarantee.

2. Identifying Path overfitting and low entropy bottlenecks. We identify a failure mode of naive path–flow alignment, which we call path overfitting. As the flow-matching loss continues to improve, generation quality initially improves but can later degrade. Sections 3.5 and 4.1 empirically associate this failure with a phenomenon that we call a low-entropy bottleneck; Proposition 2 gives an exact entropy-rate interpretation of the divergence diagnostic.

3. Stochastic Gaussian regularization with entropy guarantees. We design a simple regularization scheme based on perturbing the source input seen by the learned path network. Under independent endpoint coupling, Theorem 1 lower-bounds the entropy of the stochastic trainingpath marginals. Proposition 3 further lower bounds their instantaneous rate of entropy decrease.

4. ImageNet-scale validation and extension to model guidance. On ImageNet-256 × 256 with SiT backbones, our method consistently improves FID across model sizes while keeping sampling cost unchanged (Table 1). Our alignment template also composes with model-guidance [18] training, and shows significant improvement over the model-guidance baseline (Table 1).

## 2 Related Work

Flow matching trains continuous-time generative models by regressing a velocity field to tangents of a prescribed probability path [11]. Conditional flow matching, optimal-transport variants, and stochastic interpolants show that the path, coupling, and induced velocity target are central modeling choices [19, 2, 1]. Our work stays within this framework, and focuses on the concrete question of how to train the path. Prior work improves flow matching by changing endpoint couplings or weights [19, 15, 4, 10], straightening trajectories through reflow [12], or replacing Euclidean lines with geometric paths [8]. Closest to us are methods that learn paths or forward processes, including learned curved paths for straighter flows [17], Neural Flow Diffusion Models [3], ALI-CFM for observed multi-marginal snapshots [9], and Curly-FM for non-gradient dynamics from population snapshots [14]. Our method differs in purpose and use: after endpoints are sampled, we learn an endpoint-conditioned residual path as a training-time supervision mechanism. Rather than optimizing straightness, likelihood, or matching observed intermediate marginals, we optimize for path–flow alignment, and regularize the intermediate marginals via stochastic regularization. Classifier-free guidance modifies sampling by combining conditional and unconditional predictions [7], while model guidance folds a guided direction into training so that sampling requires a single conditional evaluation [18]. Our main experiments use standard CFG evaluation (Section 6); Section 5 show that the same path–flow alignment template also applies to model-guided velocity objectives.

## 3 Flow-Guided Interpolation

This section defines the learned training path. We first recall the population correctness statement for arbitrary endpoint-preserving paths, then introduce our residual parameterization and the alternating path–flow alignment algorithm.

## 3.1 Problem setup and notation

Let $p _ { 0 } = \mathcal { N } ( 0 , I _ { d } )$ be the latent source distribution. For each label $y \in \mathcal { V } .$ , let $p _ { 1 } ( \cdot \mid y )$ denote the target latent distribution. For simplicity, this paper considers only independently coupled $( x _ { 0 } , x _ { 1 } )$ , with joint probability $( x _ { 0 } , x _ { 1 } \mid y ) \sim p _ { 0 } ( x _ { 0 } ) p _ { 1 } ( x _ { 1 } \mid y )$ . The path–flow alignment algorithm and the population flow-matching correctness results below extend to more general endpoint couplings under their stated regularity conditions. However, the Gaussian-smoothing and entropy guarantees in Section 4.2 require an additional conditional-independence condition that is satisfied by the independent coupling above but need not hold for general optimal-transport couplings [19, 10]. For any given endpoints $( x _ { 0 } , x _ { 1 } )$ , define the standard linear interpolation

$$
\gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) = ( 1 - t ) x _ { 0 } + t x _ { 1 } , \qquad t \in [ 0 , 1 ] .\tag{1}
$$

The linear path has constant velocity $x _ { 1 } - x _ { 0 }$ . Standard conditional flow matching trains a velocity network $v _ { \phi } ( x , t , y )$ by matching ${ \dot { \gamma } } _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) = x _ { 1 } - x _ { 0 }$ at the intermediate state $\gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } )$

More generally, one can replace $\gamma _ { \mathrm { l i n } }$ by any sufficiently regular random path [11, 2, 1]. Under the regularity conditions of Proposition 1, an endpoint-preserving path induces an ideal ODE transport from source to target. We write a generic path as $\gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi )$ , where $\xi$ captures auxiliary randomness.

## 3.2 Flow matching and induced transport

Given any path $\gamma ,$ let $p _ { \gamma , t } ( \cdot \mid y )$ be the distribution of $\gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi )$ conditional on $y ,$ where $\xi$ is sampled from its prescribed auxiliary distribution. The flow-matching loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { \scriptsize { f l o w } } } ( v _ { \phi } \mid \gamma ) = \mathbb { E } _ { y , x _ { 1 } , x _ { 0 } , \xi , t } \Big [ \| v _ { \phi } \big ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y \big ) - \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{2}
$$

$\operatorname { I f } \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = \gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } )$ , then ${ \dot { \gamma } } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x _ { 1 } - x _ { 0 }$ , and $( 2 )$ reduces to the standard conditional flow-matching loss $\mathbb { E } _ { y , x _ { 1 } , x _ { 0 } , t } \Big [ \big \| v _ { \phi } \big ( \gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) , t , y \big ) - \big ( x _ { 1 } - x _ { 0 } \big ) \big \| _ { 2 } ^ { 2 } \Big ]$

We call $\gamma$ endpoint-preserving ${ \mathrm { i f } } ,$ almost surely,

$$
\gamma ( 0 ; x _ { 0 } , x _ { 1 } , y , \xi ) = x _ { 0 } , \qquad \gamma ( 1 ; x _ { 0 } , x _ { 1 } , y , \xi ) = x _ { 1 } .\tag{3}
$$

Proposition 1 (Population target and induced ODE transport). Assume

$$
\mathbb { E } \int _ { 0 } ^ { 1 } \| \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \| _ { 2 } ^ { 2 } \ d t < \infty .
$$

For any fixed path $\gamma ,$ the population minimizer of (2) is

$$
v _ { \gamma } ^ { \star } ( x , t , y ) = \mathbb { E } [ \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \mid \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x , t , y ] .\tag{4}
$$

Assume in addition that $t \mapsto \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi )$ is almost surely absolutely continuous, that $\gamma$ is endpoint-preserving, and that $x _ { 0 } \sim p _ { 0 }$ and $x _ { 1 } \sim p _ { 1 } ( \cdot \mid y )$ . For each fixed $y ,$ assume that $v _ { \gamma } ^ { \star }$ admits a jointly Borel version and,for almost every $t ,$ satisfies

$$
\begin{array} { r } { \left\| v _ { \gamma } ^ { \star } ( x , t , y ) - v _ { \gamma } ^ { \star } ( x ^ { \prime } , t , y ) \right\| _ { 2 } \leq L _ { y } ( t ) \left\| x - x ^ { \prime } \right\| _ { 2 } , } \\ { \left\| v _ { \gamma } ^ { \star } ( x , t , y ) \right\| _ { 2 } \leq A _ { y } ( t ) \big ( 1 + \left\| x \right\| _ { 2 } \big ) , } \end{array}
$$

for all $x , x ^ { \prime } \in \mathbb { R } ^ { d }$ , where $L _ { y } , A _ { y } \in L ^ { 1 } ( [ 0 , 1 ] )$ . Then the ODE

$$
\frac { d } { d t } z ( t ) = v _ { \gamma } ^ { \star } ( z ( t ) , t , y ) , \qquad z ( 0 ) \sim p _ { 0 } ,\tag{5}
$$

has a unique Carathéodoryflow whose marginals match those $o f \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi )$ for all $t \in [ 0 , 1 ]$ In particular,

$$
z ( 1 ) \sim p _ { 1 } ( \cdot \mid y ) .
$$

We defer the proof to Appendix E.1. Intuitively, (4) averages the tangents of all paths passing through the same state, time, and label. Therefore the ODE driven by $v _ { \gamma } ^ { \star }$ moves probability mass in the same way as the path marginals.

## 3.3 Endpoint-preserving neural paths

Having verified the correctness of the flow-matching objective (2) for general paths, we now discuss how to use a residual path network $R _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) : [ 0 , 1 ] \times \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \times \mathcal { V }  \mathbb { R } ^ { d }$ . to parameterize a general learnable path:

$$
\gamma _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) = \gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) + t ( 1 - t ) R _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) .\tag{6}
$$

We describe the detailed implementation of $R _ { \psi }$ in Appendix $\mathrm { C } ;$ all our results are agnostic to the parameterization of $R _ { \psi }$ . The envelope $t ( 1 - t )$ allows the residual to curve the interior of the path while enforcing endpoint preservation (3). This envelope has previously been used in [17, 14, 3]. By observing that $\mathbf { \bar { \Psi } } t ( 1 - t ) = 0$ at t = 0 and $t = 1$ , we immediately verify that

$$
\gamma _ { \psi } ( 0 ; x _ { 0 } , x _ { 1 } , y ) = x _ { 0 } , \qquad \gamma _ { \psi } ( 1 ; x _ { 0 } , x _ { 1 } , y ) = x _ { 1 } .\tag{7}
$$

## 3.4 Flow-guided path alignment and alternating optimization

Path and flow are trained on the same objective. For a fixed path $\gamma _ { \psi }$ , the flow phase minimizes

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { \hat { H } o w } } ( \phi \mid \psi ) = \mathbb { E } _ { y , x _ { 1 } , x _ { 0 } , t } \Big [ \left\| v _ { \phi } ( \gamma _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) , t , y ) - \dot { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) \right\| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{8}
$$

The path phase uses the same objective, but freezes $v _ { \phi }$ and optimizes $\gamma _ { \psi }$

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { p a t h } } \big ( \psi \mid \phi \big ) = \mathbb { E } _ { y , x _ { 1 } , x _ { 0 } } \int _ { 0 } ^ { 1 } \big \| v _ { \phi } \big ( \gamma _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) , t , y \big ) - \dot { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) \big \| _ { 2 } ^ { 2 } d t } } \\ & { } & { + \beta \mathbb { E } _ { y , x _ { 1 } , x _ { 0 } } \int _ { 0 } ^ { 1 } \| \dot { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) \| _ { 2 } ^ { 2 } d t , \qquad \beta \ge 0 . } \end{array}\tag{9}
$$

The first term of $( 9 )$ is identical to (8). Thus the path phase does not introduce a separate supervision signal; it uses the current flow model to align the path. The second term of (9) is the kinetic energy of the path, scaled by $\beta .$ This term regularizes the path by driving it toward the straight line; it is also important later for controlling the entropy of the induced path marginals (see Proposition 3 of Section 4.2).

We formally state the procedure in Algorithm 1 alternates between a path training phase and a flow training phase. In the simple case of $\rho _ { t } \equiv 1 , \hat { x } _ { 0 } ( t ) = x _ { 0 }$ , and Algorithm 1 is ordinary alternating minimization: the path phase aligns $\gamma _ { \psi }$ to the current flow, and the flow phase trains $v _ { \phi }$ on the current path. The optional stochastic-regularization case $( \rho _ { t } < 1 )$ is discussed later in Section 4.2.

Figure 1 illustrates the effect of path–flow alignment on the learned flow in a two-dimensional mixture example. Compared with standard flow matching, the learned flows follow more curved ODE trajectories; stochastic regularization also changes the intermediate marginals. The path network influences these trajectories by changing the states and tangent targets used to train the velocity field, while sampling uses only the learned velocity field. Appendix F.4 gives the toy setup and further discussion.

Correctness of flow-matching objective. Proposition 1 and (7) together imply that every updated path remains an exact interpolation from $p _ { 0 }$ to $p _ { 1 } ( \cdot \mid y )$ , and its population flow-matching target induces the corresponding correct ODE transport. Therefore, after each path update, the subsequent flow network phase is still a valid flow-matching problem for the current path.

## 3.5 Path overfitting

We first evaluate whether alternating optimization of $\gamma _ { \psi }$ and $v _ { \phi }$ improves the downstream flow. Figure 2 shows the flow-matching loss after successive cycles of Algorithm 1, and Figure 3 shows the corresponding FID. We ablate the kinetic energy regularization strength β. For reference, the figure also includes the stochastic source-regularized variant introduced later in Section 4. Appendix D provides additional scatter-plot diagnostics for the same SiT-S sweep, relating FID to late divergence, average speed, and average acceleration of both the learned flow and the learned path.

The two plots reveal a clear path-overfitting effect. In the $\rho _ { t } \equiv 1$ Algorithm 1 runs, the flow-matching loss continues to decrease as training proceeds. This means that the learned path and the learned flow become increasingly well aligned under the training objective. However, the corresponding FID does not continue to improve. For the unregularized runs, FID initially improves after the first cycle, but then rapidly deteriorates even as ${ \mathcal { L } } _ { \mathrm { { f l o w } } }$ keeps decreasing.

Algorithm 1 Alternating path–flow alignment with optional stochastic regularization   
Require: Path parameters $\psi ,$ flow parameters ϕ, cycles C, path steps $K _ { \psi } .$ , flow steps $K _ { \phi } ,$ , kinetic   
weight $\beta ,$ optional perturbation schedule $t \mapsto \rho _ { t }$   
1: for $c = 1 , \ldots , C$ do   
2: for $k \doteq 1 , \ldots , K _ { \psi }$ do   
3: Sample $y , \bar { x } _ { 1 } \stackrel { \triangledown } { \sim } p _ { 1 } ( \cdot \mid y ) , x _ { 0 } \sim p _ { 0 } , t \sim \mathrm { U n i f } [ 0 , 1 ] , \epsilon \sim \mathcal { N } ( 0 , I _ { d } )$   
4: $\hat { x } _ { 0 } ( t ) = \rho _ { t } x _ { 0 } + \sqrt { 1 - \rho _ { t } ^ { 2 } } \epsilon$   
5: $\begin{array} { r } { \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = ( \dot { 1 } - t ) x _ { 0 } + t x _ { 1 } + t ( 1 - t ) R _ { \psi } ( t ; \hat { x } _ { 0 } ( t ) , x _ { 1 } , y ) . } \end{array}$   
6: $\begin{array} { r } { \mathcal { I } _ { \psi } \gets \| \partial _ { t } \widehat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) - v _ { \phi } ( \widehat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) , t , y ) \| _ { 2 } ^ { 2 } + \beta \| \partial _ { t } \widehat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) \| _ { 2 } ^ { 2 } } \end{array}$   
7: Update ψ usin $\underline { { \ u { \ d } } } - \nabla _ { \psi } \mathcal { I } _ { \psi }$ , with ϕ frozen   
8: end for   
9: for $k = 1 , \ldots , K _ { \phi }$ do   
10: Sample $y , \overset { \cdot } { x } _ { 1 } \sim p _ { 1 } ( \cdot \mid y ) , x _ { 0 } \sim p _ { 0 } , t \sim \mathrm { U n i f } [ 0 , 1 ] , \epsilon \sim \mathcal { N } ( 0 , I _ { d } )$   
11: $\hat { x } _ { 0 } ( t ) = \rho _ { t } x _ { 0 } + \sqrt { 1 - \rho _ { t } ^ { 2 } } \epsilon$   
12: $\begin{array} { r } { \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = ( \dot { 1 } - t ) x _ { 0 } + t x _ { 1 } + t ( 1 - t ) R _ { \psi } ( t ; \hat { x } _ { 0 } ( t ) , x _ { 1 } , y ) . } \end{array}$   
13: $\mathcal { I } _ { \phi } \gets \lVert \partial _ { t } \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) - v _ { \phi } ( \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) , t , y ) \rVert _ { 2 } ^ { 2 }$   
14: Update ϕ usin $\begin{array} { r } { \mathbf { \mu } _ { \mathbf { \mu } } - \nabla _ { \phi } \mathcal { I } _ { \phi } , } \end{array}$ with ψ frozen   
15: end for   
16: end for

![](images/adc134967f515fc3d159710c286ee56877a0949600c007a4af0090d4772fa348.jpg)

![](images/703d66a21de933be681a523ed0e980e221066608c28e6544c5c0b08513e5efd5.jpg)  
Figure 2: Flow-matching loss across alternating training cycles on the SiT-S model.  
Figure 3: FID across alternating training cycles on the SiT-S model.

This demonstrates that a low flow-matching loss is not, by itself, a sufficient condition for a good learned path. Proposition 1 guarantees that the population target for an endpoint-preserving path induces a correct transport, but the finite trained network can still be poor for generation in practice. In other words, the path objective can overfit: it can improve the local regression problem without preserving the distributional properties needed for robust sampling.

## 4 Entropy Bottlenecks and Stochastic Regularization

Section 3.5 shows that endpoint correctness and low flow-matching loss do not by themselves guarantee good generation. The learned path can become well aligned with the current flow on its own training states while the resulting sampler develops unstable distributional behavior. In this section, we identify this behavior as an entropy bottleneck and introduce a stochastic regularizer motivated by this diagnosis.

The key diagnostic is average flow divergence. Along any probability flow, average divergence equals the rate of change of differential entropy. Thus a late positive-divergence spike means that the sampler is rapidly expanding entropy near the endpoint. If this expansion compensates for earlier

![](images/454c277f056d032efd38f620f81c23ea9d2fdeeb3b5622ee0bf323d7588ce240.jpg)  
Figure 4: Per-dimensional binned divergence of the learned SiT-S flow network along its induced probability flow.

![](images/7128be8a980d97016f4665c8309e1f1f3a841f5cc336a1f332d228b5cd862b20.jpg)  
Figure 5: Cumulative normalized divergence. Values below zero indicate an entropy bottleneck relative to the generated endpoint distribution.

over-contraction, the flow must have passed through an overly concentrated intermediate marginal.   
We call this a low-entropy bottleneck.

We next introduce stochastic source perturbation: by perturbing the $x _ { 0 }$ input to the residual path network, we effectively convolve the intermediate path marginals by an independent Gaussian, which in turn lower bounds entropy. The same smoothing also bounds Fisher information; combined with path kinetic energy, this gives an instantaneous lower bound on entropy decrease (equivalently average flow divergence).

To keep the notation separate, Section 4.1 diagnoses the probability flow generated by the learned sampler $v _ { \phi }$ , whose marginals are denoted $p _ { \phi , t }$ . Section 4.2 analyzes the stochastic-regularized training path $\hat { \gamma } _ { \psi } .$ , whose marginals are denoted $\hat { p } _ { \psi , t }$ . In the population limit where the trained flow equals the path-induced target, $v _ { \phi } = \hat { v } _ { \psi } ^ { \star }$ , the sampler marginals $p _ { \phi , t }$ coincide with the regularized path marginals $\hat { p } _ { \psi , t }$

The distinction between these marginals is important for the scope of our results. Theorem 1 and Proposition 3 below control the stochastic training-path marginals $\hat { p } _ { \psi , t } ,$ , whereas Proposition 2 applies directly to the finite learned-sampler marginals $p _ { \phi , t }$ . The two are guaranteed to coincide when $v _ { \phi } = \hat { v } _ { \psi } ^ { \star }$ , but may differ under finite optimization and model capacity. For a fixed path, population excess flow-matching risk controls an $L ^ { 2 }$ velocity discrepancy, while entropy evolution depends on velocity derivatives through divergence; velocity-level control alone therefore does not transfer the path-level entropy guarantee to the learned sampler. We accordingly interpret the divergence measurements below as empirical evidence about the learned sampler rather than as a consequence of the path-level guarantee.

## 4.1 Divergence spikes reveal entropy bottlenecks

An important diagnostic for the path-overfitting phenomenon in Section 3.5 is the average divergence of the learned velocity field $v _ { \phi } .$ , which has a direct connection to entropy: for any probability flow, the expected divergence of the velocity field equals the time derivative of differential entropy. Thus a large positive-divergence spike near the endpoint indicates that the learned flow is rapidly increasing entropy at late time. If this late expansion follows an earlier over-contraction, then the probability flow has passed through a low-entropy bottleneck.

For each trained model, let $z _ { \phi } ( t )$ denote the ODE trajectory generated by the learned flow,

$$
\frac { d } { d t } z _ { \phi } ( t ) = v _ { \phi } ( z _ { \phi } ( t ) , t , y ) , \qquad z _ { \phi } ( 0 ) \sim p _ { 0 } ,\tag{10}
$$

and let $p _ { \phi , t } ( \cdot \mid y )$ be the law of $z _ { \phi } ( t )$ conditional on y. We divide the time interval [0, 1] into ten bins. For a bin [a, b], we estimate the expected flow divergence

$$
\bar { D } _ { \phi } ( [ a , b ] ) = \frac { 1 } { d ( b - a ) } \int _ { a } ^ { b } \mathbb { E } _ { y } \mathbb { E } _ { x \sim p _ { \phi , t } ( \cdot | y ) } \left[ \nabla _ { x } \cdot v _ { \phi } ( x , t , y ) \right] d t ,\tag{11}
$$

using a Hutchinson trace estimator. The factor $1 / d$ gives the per-dimensional divergence. Figure 4 plots this expected flow divergence against the time, for flows $v _ { \phi }$ trained using baseline flowmatching, as well as using Algorithm 1 with different values of $\rho _ { t }$ . The overfit $\rho _ { t } = 1$ run produces a pronounced positive-divergence spike near $t = 1$ , in sharp contrast to the baseline and the stochastic source-regularized run introduced in Section 4.2.

The following proposition relates flow-divergence to entropy-change:

Proposition 2 (Expected divergence equals entropy rate). Let $p _ { t } ( \cdot \mid y )$ evolve under the continuity equation $\partial _ { t } p _ { t } ( x \mathbin { | } \mathbin { \mathit { \check { y } } } ) + \nabla _ { x } \cdot ( \bar { p _ { t } } ( x \mathbin { | } \mathbin { \mathit { y } } ) ) \bar { u } _ { t } ( x , y ) ) \mathbin { \mathop = } 0$ , with sufficient smoothness, decay at infinity, and finite differential entropy. Equivalently, when the ODE is well posed, $p _ { t } ( \cdot \mid y )$ is the law ofthe solution to $\begin{array} { r } { \frac { d } { d t } z ( t ) = u _ { t } ( z ( t ) , y ) } \end{array}$ . Then

$$
\frac { d } { d t } h ( p _ { t } ( \cdot \mid y ) ) = \mathbb { E } _ { x \sim p _ { t } ( \cdot \mid y ) } \left[ \nabla _ { x } \cdot u _ { t } ( x , y ) \right] .\tag{12}
$$

Consequently, for any time bin $0 \leq a < b \leq 1$

$$
\frac { 1 } { b - a } \int _ { a } ^ { b } \mathbb { E } _ { x \sim p _ { t } ( \cdot \vert y ) } \left[ \nabla _ { x } \cdot u _ { t } ( x , y ) \right] d t = \frac { h ( p _ { b } ( \cdot \vert y ) ) - h ( p _ { a } ( \cdot \vert y ) ) } { b - a } .\tag{13}
$$

We defer the proof to Appendix E.2. We apply Proposition 2 directly to the learned flow by taking $u _ { t } ( \cdot , y ) = v _ { \phi } ( \cdot , t , y ) , p _ { t } ( \cdot \mid y ) = p _ { \phi , t } ( \cdot \mid y )$ . This gives the exact identity

$$
\frac { d } { d t } h ( p _ { \phi , t } ( \cdot \mid y ) ) = \mathbb { E } _ { x \sim p _ { \phi , t } ( \cdot \mid y ) } \left[ \nabla _ { x } \cdot \boldsymbol { v } _ { \phi } ( x , t , y ) \right] .\tag{14}
$$

Thus the divergence curves in Figure 4 are not merely local geometric measurements of $v _ { \phi } ;$ they estimate the entropy-rate profile of the probability flow generated by the learned sampler. If, in the population limit, $v _ { \phi }$ equals the ideal path-induced velocity $v _ { \psi } ^ { \star }$ from Proposition 1, then $p _ { \phi , t } = p _ { \psi , t }$ by the induced ODE transport statement, and the generated endpoint is $p _ { 1 } ( \cdot \mid y )$

The identity also gives a precise meaning to a low-entropy bottleneck. For any $s < 1$ , applying (14) on [s, 1] gives $\begin{array} { r } { h ( p _ { \phi , 1 } ( \cdot \mid y ) ) - h ( p _ { \phi , s } ( \cdot \mid y ) ) = \int _ { s } ^ { 1 } \mathbb { E } _ { x \sim p _ { \phi , t } ( \cdot \mid y ) } \left[ \nabla _ { x } \cdot v _ { \phi } ( x , t , y ) \right] d t } \end{array}$ . Therefore,

$$
\int _ { s } ^ { 1 } \mathbb { E } _ { { x } \sim { p _ { \phi , t } ( \cdot \vert y ) } } \left[ \nabla _ { x } \cdot v _ { \phi } ( x , t , y ) \right] d t > 0 \quad \Longleftrightarrow \quad h ( p _ { \phi , s } ( \cdot \vert y ) ) < h ( p _ { \phi , 1 } ( \cdot \vert y ) ) .\tag{15}
$$

We call this event a low-entropy bottleneck: the learned probability flow has contracted below the entropy of its own endpoint distribution $p _ { \phi , 1 }$ and must subsequently expand entropy before $t = 1$ In the ideal endpoint-preserving case $p _ { \phi , 1 } = p _ { 1 }$ , so the same condition says that the intermediate marginal has entropy below the target entropy.

Figure 5 visualizes this effect through cumulative divergence. Let $\bar { D } _ { \phi } ( t )$ denote the piecewise-constant, per-dimensional binned estimate from (11). We plot $\widehat { G } _ { \phi } ( t ) ~ = ~ 1 -$ $( \int _ { 0 } ^ { t } \bar { D } _ { \phi } ( r ) d r ) / ( \int _ { 0 } ^ { 1 } \bar { D } _ { \phi } ( r ) d r )$ . For the exact entropy-rate curve, this quantity equals the normalized excess entropy above the endpoint,

$$
G _ { \phi } ( t ; y ) = \frac { h ( p _ { \phi , t } ( \cdot \mid y ) ) - h ( p _ { \phi , 1 } ( \cdot \mid y ) ) } { h ( p _ { 0 } ) - h ( p _ { \phi , 1 } ( \cdot \mid y ) ) } = 1 - \frac { \int _ { 0 } ^ { t } \mathbb { E } _ { x \sim p _ { \phi , r } ( \cdot \mid y ) } [ \nabla _ { x } \cdot v _ { \phi } ( x , r , y ) ] d r } { \int _ { 0 } ^ { 1 } \mathbb { E } _ { x \sim p _ { \phi , r } ( \cdot \mid y ) } [ \nabla _ { x } \cdot v _ { \phi } ( x , r , y ) ] d r } .\tag{16}
$$

The plotted statistic estimates the divergences with Hutchinson traces. A value near 1 corresponds to source-side entropy, a value near 0 corresponds to reaching the endpoint entropy, and a value below 0 indicates entropy below the endpoint entropy. The overfit run dips below zero around $t \approx 0 . 9$ matching the large positive-divergence spike in Figure 4. This supports the interpretation that the overfit flow passes through a low-entropy bottleneck and then requires a compensating expansion.

## 4.2 Regularization via Gaussian perturbation

Section 4.1 suggests that path overfitting is associated with entropy bottlenecks: the learned path can reduce its local regression objective while routing probability mass through overly concentrated intermediate marginals. We therefore regularize the residual path network by hiding part of the source noise from the residual input. The key design choice is that the perturbation is applied only to the input of $R _ { \psi } ;$ the endpoint interpolation itself still uses the true endpoints $( x _ { 0 } , x _ { 1 } )$

Let $\rho _ { t } \in [ 0 , 1 ]$ be a deterministic perturbation schedule and let $\epsilon \sim \mathcal { N } ( 0 , I _ { d } )$ be independent of $( x _ { 0 } , x _ { 1 } , y )$ . Define

$$
\hat { x } _ { 0 } ( t ) = \rho _ { t } x _ { 0 } + \sqrt { 1 - \rho _ { t } ^ { 2 } } \epsilon .\tag{17}
$$

For each fixed $t ,$ the perturbed source input is still marginally standard Gaussian, $\hat { x } _ { 0 } ( t ) \sim \mathcal { N } ( 0 , I _ { d } )$ but its correlation with the true source endpoint is reduced from 1 to $\rho _ { t }$ . Thus the residual network receives a source-like input with less information about the particular endpoint noise $x _ { 0 }$ that appears in the actual interpolation. The stochastic-regularized path is

$$
\begin{array} { r } { \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = ( 1 - t ) x _ { 0 } + t x _ { 1 } + t ( 1 - t ) R _ { \psi } ( t ; \hat { x } _ { 0 } ( t ) , x _ { 1 } , y ) . } \end{array}\tag{18}
$$

This is the stochastic path used in Algorithm 1. Setting $\rho _ { t } \equiv 1$ gives $\hat { x } _ { 0 } ( t ) = x _ { 0 }$ and recovers the unregularized path $\gamma _ { \psi }$ . Setting $\rho _ { t } < 1$ hides part of $x _ { 0 }$ from the residual network, while leaving the true linear endpoint term unchanged. When $\rho _ { t }$ is time-dependent, the tangent $\partial _ { t } \hat { \gamma } _ { \psi }$ differentiates through both the envelope $t ( 1 - t )$ and the schedule $\rho _ { t }$ in $( 1 7 )$ . The perturbation does not change the endpoints. For every schedule $\rho _ { t }$ , every $\epsilon ,$ and every residual network $R _ { \psi }$

$$
\hat { \gamma } _ { \psi } ( 0 ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = x _ { 0 } , \qquad \hat { \gamma } _ { \psi } ( 1 ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = x _ { 1 } .\tag{19}
$$

Only the residual term sees the corrupted source input, and this residual term is multiplied by $t ( 1 - t )$ which vanishes at both endpoints. When path learning is disabled, $R _ { \psi } \equiv 0 , \mathrm { { s o } } \left( 1 8 \right)$ reduces to the standard linear path independently of $\rho _ { t } ;$ ; stochastic source perturbation therefore has no standalone effect without a learned residual path.

Let $\hat { p } _ { \psi , t } ( \cdot \mid  { y } )$ denote the distribution of $\hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon )$ conditional on $y ,$ where $( x _ { 0 } , x _ { 1 } )$ is sampled from the chosen endpoint coupling and $\epsilon \sim \mathcal { N } ( 0 , I _ { d } )$ . The corresponding population flow-matching target is

$$
\begin{array} { r } { \hat { v } _ { \psi } ^ { \star } ( x , t , y ) = \mathbb { E } \left[ \partial _ { t } \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) ~ | ~ \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = x , t , y \right] . } \end{array}\tag{20}
$$

Thus Gaussian perturbation changes the training path distribution, but it does not make the flowmatching target biased or approximate: the population target remains the exact marginal velocity of the perturbed path.

Corollary 1 (Correctness of stochastic source perturbation). Assume that the stochastic path (18) satisfies the assumptions of Proposition 1. Then the flow-matching loss trained on pairs $\big ( \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) , \partial _ { t } \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) \big )$ has population minimizer (20). Moreover, the probability flow generated by $\hat { v } _ { \psi } ^ { \star }$ transports p<sub>0</sub> $t o p _ { 1 } ( \cdot \mid y )$

Proof. The additional Gaussian variable ϵ is simply auxiliary randomness in the sense of Section 3.2.   
Endpoint preservation follows from (19). The Corollary follows immediately from Proposition 1.

We now explain why the stochastic source perturbation in Section 4.2 directly regularizes the intermediate marginals: First, at any fixed time t away from the endpoint, the part of $x _ { 0 }$ hidden from the residual network can be rewritten as independent additive Gaussian noise in the state. Thus $\hat { p } _ { \psi , t } ( { \cdot } | \ y )$ is a Gaussian-smoothed distribution and has an explicit entropy floor. Second, the same smoothing gives a Fisher-information bound. Combining this Fisher-information bound with the pointwise kinetic energy of the path yields an instantaneous lower bound on the entropy rate, or equivalently on the average divergence of the ideal marginal velocity.

The correctness results above do not require the independent endpoint coupling. For the Gaussiansmoothing result below, however, we use the independent coupling from Section 3.1: conditional on $y ,$ $x _ { 0 } \sim \mathcal { N } ( 0 , I _ { d } )$ and $x _ { 1 } \sim p _ { 1 } ( \cdot \mid y )$ are independent, and $\epsilon \sim \mathcal { N } ( 0 , I _ { d } )$ is independent of both. More generally, a sufficient condition for the argument below is that $\eta _ { t }$ in (23) is conditionally standard Gaussian and independent of $( \hat { x } _ { 0 } ( t ) , x _ { 1 } )$ given y. A general optimal-transport coupling need not satisfy this condition.

Theorem 1 (Source randomization induces Gaussian smoothing). Fix a time $t \in [ 0 , 1 )$ such that $\rho _ { t } < 1$ , and define

$$
\sigma _ { t } = ( 1 - t ) \sqrt { 1 - \rho _ { t } ^ { 2 } } .\tag{21}
$$

Let $\hat { p } _ { \psi , t } ( { \cdot } | \ y )$ be the law ofthe stochastic path (18) conditional on y. Then there exists a distribution $q _ { \psi , t } ( \cdot \mid y )$ such that

$$
\hat { p } _ { \psi , t } ( \cdot \mid y ) = q _ { \psi , t } ( \cdot \mid y ) * \mathcal { N } ( 0 , \sigma _ { t } ^ { 2 } I _ { d } ) .\tag{22}
$$

More explicitly, if

$$
\eta _ { t } = \frac { x _ { 0 } - \rho _ { t } \hat { x } _ { 0 } ( t ) } { \sqrt { 1 - \rho _ { t } ^ { 2 } } } ,\tag{23}
$$

then, conditional on $y , \eta _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$ and $\eta _ { t }$ is independent of $( \hat { x } _ { 0 } ( t ) , x _ { 1 } )$ , and

$$
\begin{array} { r } { \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = ( 1 - t ) \rho _ { t } \hat { x } _ { 0 } ( t ) + t x _ { 1 } + t ( 1 - t ) R _ { \psi } ( t ; \hat { x } _ { 0 } ( t ) , x _ { 1 } , y ) + \sigma _ { t } \eta _ { t } . } \end{array}\tag{24}
$$

The distribution $q _ { \psi , t } ( \cdot \mid y )$ in (22) is the law, conditional on y, ofthe non-Gaussian term

$$
( 1 - t ) \rho _ { t } { \hat { x } } _ { 0 } ( t ) + t x _ { 1 } + t ( 1 - t ) R _ { \psi } ( t ; { \hat { x } } _ { 0 } ( t ) , x _ { 1 } , y ) .\tag{25}
$$

Consequently, whenever the differential entropy is well-defined

$$
h ( \hat { p } _ { \psi , t } ( \cdot \mid y ) ) \geq \frac { d } { 2 } \log \bigl ( 2 \pi e \sigma _ { t } ^ { 2 } \bigr ) = \frac { d } { 2 } \log \bigl ( 2 \pi e ( 1 - t ) ^ { 2 } ( 1 - \rho _ { t } ^ { 2 } ) \bigr ) .\tag{26}
$$

Moreover, the Fisher information ofthe smoothed marginal satisfies

$$
I ( \hat { p } _ { \psi , t } ( \cdot \vert y ) ) : = \mathbb { E } _ { x \sim \hat { p } _ { \psi , t } ( \cdot \vert y ) } \left[ \Vert \nabla _ { x } \log \hat { p } _ { \psi , t } ( x \mid y ) \Vert _ { 2 } ^ { 2 } \right] \leq \frac { d } { \sigma _ { t } ^ { 2 } } = \frac { d } { ( 1 - t ) ^ { 2 } ( 1 - \rho _ { t } ^ { 2 } ) } .\tag{27}
$$

We defer the proof to Appendix E.3. The entropy floor in (26) is a state-level obstruction to arbitrarily deep entropy collapse. The bound becomes vacuous as $t  1$ , which is unavoidable: endpoint preservation requires the added smoothing to vanish at the target endpoint.

This smoothing result also gives the bin-level intuition behind the divergence diagnostics above. Applying Proposition 2 to the ideal velocity $\hat { v } _ { \psi } ^ { \star }$ gives, for any $0 \leq a < b < 1$

$$
\int _ { a } ^ { b } \mathbb { E } _ { x \sim \hat { p } _ { \psi , t } ( \cdot | y ) } [ \nabla _ { x } \cdot \hat { v } _ { \psi } ^ { \star } ( x , t , y ) ] d t = h ( \hat { p } _ { \psi , b } ( \cdot | y ) ) - h ( \hat { p } _ { \psi , a } ( \cdot | y ) ) .
$$

Thus binned average divergence is a finite-difference entropy rate. Since (26) lower-bounds the entropy at the end of any bin with $\sigma _ { b } > 0$ , Gaussian perturbation prevents arbitrarily deep entropy collapse on such a bin unless the entropy at the beginning of the bin is correspondingly large.

We next give an instantaneous version of this control:

Proposition 3 (Entropy-rate lower bound under kinetic-energy control). Assume the regularity conditions needed for Proposition 2 and the coupling and regularity conditions of Theorem 1. Define the pointwise stochastic-path kinetic energy

$$
L _ { \mathrm { e u c } } ^ { \rho , \psi } ( t , y ) : = \mathbb { E } \left[ \left\| \partial _ { t } \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) \right\| _ { 2 } ^ { 2 } \middle | y \right] ,\tag{28}
$$

where the superscript ρ denotes the schedule $t \mapsto \rho _ { t }$ . Then, for every $t < 1$ with $\rho _ { t } < 1$

$$
\frac { d } { d t } h ( \hat { p } _ { \psi , t } ( \cdot \mid y ) ) \geq - \frac { \sqrt { d L _ { \mathrm { e u c } } ^ { \rho , \psi } ( t , y ) } } { ( 1 - t ) \sqrt { 1 - \rho _ { t } ^ { 2 } } } = - \frac { d } { 1 - t } \sqrt { \frac { L _ { \mathrm { e u c } } ^ { \rho , \psi } ( t , y ) } { d ( 1 - \rho _ { t } ^ { 2 } ) } } .\tag{29}
$$

We defer the proof to Appendix E.4. Proposition 3 shows that stochastic perturbation and kinetic regularization play complementary roles. The smoothing scale $\sigma _ { t } = ( 1 - t ) \sqrt { 1 - \rho _ { t } ^ { 2 } }$ controls the Fisher information of the marginal: larger smoothing makes the score smaller and prevents sharper entropy collapse. The kinetic term controls the size of the marginal velocity acting on this smoothed distribution. Thus $L _ { \mathrm { e u c } }$ is not only a geometric straightness penalty; in the presence of Gaussian perturbation, it also bounds how fast the path-induced marginal can decrease entropy.

The scaling is comparable to the standard linear path. If $R _ { \psi } \equiv 0$ and the target is a point mass, then the linear path has marginal $\mathcal { N } ( 0 , ( 1 - t ) ^ { 2 } I _ { d } )$ and entropy rate exactly $- d / ( 1 - t )$ . The bound (29) recovers this order when $\rho _ { t }$ is bounded away from 1. Conversely, when $\rho _ { t }$ is close to 1, there is little smoothing and the bound appropriately becomes weak.

## 5 Generalizing Beyond Flow Matching

Path–flow alignment is not tied to the standard flow-matching velocity; it only needs a velocity-like target that is regressed to the path tangent. Model guidance [18] provides such a target by folding classifier-free guidance into training. For guidance scale ω, define

$$
\begin{array} { r } { \tilde { v } _ { \phi } ^ { \omega } ( x , t , y ) = v _ { \phi } ( x , t , y ) - \omega \mathrm { s g } ( v _ { \phi } ( x , t , y ) - v _ { \phi } ( x , t , \emptyset ) ) , } \end{array}
$$

where sg denotes stop-gradient, and train with

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M G } } ( \phi \mid \gamma ) = \mathbb { E } _ { y , x _ { 1 } , x _ { 0 } , \xi , t } \left\| \tilde { v } _ { \phi } ^ { \omega } ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y ) - \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{30}
$$

We use the same alternating updates as Algorithm 1, replacing $v _ { \phi }$ by $\tilde { v } _ { \phi } ^ { \omega }$ in both phases. Stochastic source perturbation and endpoint preservation are unchanged. Table 1 shows that this extension improves 50k-FID over standard MG fine-tuning on both SiT-B-MG and SiT-XL-MG.

## 6 Experiments

Unless otherwise stated, we use $\beta = 0 . 0 5$ and $\rho _ { t } = c ( 1 - t )$ with c = 1 for SiT-S/B/L and $c = 0 . 7$ for SiT-XL. We provide full training setups in Appendix F.

ImageNet results. We evaluate regularized path–flow alignment on class-conditional ImageNet-256 × 256 using SiT backbones [13, 5]. Each run starts from the same pretrained SiT checkpoint. The baseline continues standard flow-matching training for 12k steps; our method uses three alternating cycles, each with 3k path updates followed by 4k flow updates. Table 1 reports the best 50k-FID achieved within the 12k flow-update budget. Our method improves FID at every scale, including SiT-XL, where standard fine-tuning does not improve the starting checkpoint. The largest gains are on SiT-L and SiT-XL, with relative FID improvements of 19.1% and 12.7%, respectively. The full checkpoint-wise FID/IS table is in Appendix Table 3.

Model guidance. We also test the effective-velocity extension from Section 5. Starting from FID converged SiT-B-MG and SiT-XL-MG checkpoints, we compare standard MG fine-tuning baseline against path–flow alignment with the MG objective. Table 1 shows that our method improves the best FID from 7.940 to 5.585 on SiT-B-MG and from 1.494 to 1.322 on SiT-XL-MG. Checkpoint-wise FID results are in Appendix Table 4.

The FID improvements persist against compute-matched continued finetuning (Appendix F.1) and across three disjoint generation-seed sets (Appendix F.2). The ImageNet results improve recall and Inception Score (Appendix F.2 and Table 3). We also find gains on FlowDCN-B (Appendix F.3) and compare with [17] (Appendix F.3). We demonstrate that path–flow alignment increases curvature of sampled trajectories on a toy setup and include qualitative comparisons showing improvements in image quality (Appendix F.4).

Table 1: ImageNet- $2 5 6 \times 2 5 6$ summary. We report the best 50k-FID (↓) over the same 12k continuation budget; full checkpoint-wise FID/IS results are in Appendix Tables 3–4. Params denote the inference-time SiT flow model only; the path network is used only during training.
<table><tr><td>Objective</td><td>Model</td><td>Params</td><td>Init FID</td><td>Baseline best</td><td>Ours best</td><td>FID gain</td></tr><tr><td>FM</td><td>SiT-S</td><td>33M</td><td>9.109</td><td>9.022</td><td>8.359</td><td>7.3%</td></tr><tr><td>FM</td><td>SiT-B</td><td>130M</td><td>5.074</td><td>5.057</td><td>4.415</td><td>12.7%</td></tr><tr><td>FM</td><td>SiT-L</td><td>458M</td><td>3.424</td><td>3.390</td><td>2.742</td><td>19.1%</td></tr><tr><td>FM</td><td>SiT-XL</td><td>675M</td><td>2.091</td><td>2.091</td><td>1.825</td><td>12.7%</td></tr><tr><td>MG</td><td>SiT-B-MG</td><td>130M</td><td>8.090</td><td>7.940</td><td>5.585</td><td>29.7%</td></tr><tr><td>MG</td><td>SiT-XL-MG</td><td>675M</td><td>1.522</td><td>1.494</td><td>1.322</td><td>11.5%</td></tr></table>

## 7 Conclusion

We introduced path–flow alignment, which jointly learns an endpoint-preserving training path and flow model, and identified path overfitting as a failure mode of naive joint optimization. Stochastic source perturbation provides Gaussian-smoothing and entropy guarantees for the learned training path, while our experiments show that it also suppresses the associated late-time divergence bottleneck in finite learned samplers. Across SiT and FlowDCN models, the resulting post-training procedure improves generation quality under compute-matched comparisons without changing the inference architecture or sampler. Our theory does not establish a causal link between entropy bottlenecks and generation quality, nor does it guarantee transfer of the path-level entropy control to an imperfectly fitted flow; understanding this connection and extending joint path–flow training beyond pretrained image generators remain important directions.

## Funding and Competing Interests

The authors received no third-party funding or support for this work, and declare no competing financial interests.

## References

[1] Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions, 2025.

[2] Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants, 2023.

[3] Grigory Bartosh, Dmitry Vetrov, and Christian A. Naesseth. Neural flow diffusion models: Learnable forward process for improved diffusion modelling, 2025.

[4] Sergio Calvo-Ordonez, Matthieu Meunier, Alvaro Cartea, Christoph Reisinger, Yarin Gal, and Jose Miguel Hernandez-Lobato. Weighted conditional flow matching, 2026.

[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.

[6] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018.

[7] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022.

[8] Kacper Kapusniak, Peter Potaptchik, Teodora Reu, Leo Zhang, Alexander Tong, Michael´ Bronstein, Avishek Joey Bose, and Francesco Di Giovanni. Metric flow matching for smooth interpolations on the data manifold, 2024.

[9] Oskar Kviman, Kirill Tamogashev, Nicola Branchini, Víctor Elvira, Jens Lagergren, and Nikolay Malkin. Multi-marginal flow matching with adversarially learnt interpolants, 2026.

[10] Yexiong Lin, Yu Yao, and Tongliang Liu. Beyond optimal transport: Model-aligned coupling for flow matching, 2025.

[11] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023.

[12] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022.

[13] Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers, 2024.

[14] Katarina Petrovic, Lazar Atanackovic, Viggo Moro, Kacper Kapu ´ sniak, ´ <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, Michael Bronstein, Avishek Joey Bose, and Alexander Tong. Curly flow matching for learning non-gradient field dynamics, 2025.

[15] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky T. Q. Chen. Multisample flow matching: Straightening flows with minibatch couplings, 2023.

[16] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans, 2016.

[17] Shiv Shankar and Tomas Geffner. Learning straight flows by learning curved interpolants, 2025.

[18] Zhicong Tang, Jianmin Bao, Dong Chen, and Baining Guo. Diffusion models without classifierfree guidance, 2025.

[19] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport, 2024.

[20] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Flowdcn: Exploring dcn-like architectures for fast image generation with arbitrary resolution, 2024.

## A Notation guide

Table 2 summarizes the main path, marginal, and velocity notation used throughout the paper. The generic notation $\gamma$ and $v _ { \gamma } ^ { \star }$ is used in the population flow-matching statements; the symbols $\gamma _ { \psi }$ and $\hat { \gamma } _ { \psi }$ denote the two learned-path special cases used by the algorithm and the stochastic regularizer.

Table 2: Notation convention for paths, induced marginals, and velocity fields.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\gamma _ { \mathrm { l i n } }$ </td><td>Standard linear path,  $\gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ </td></tr><tr><td> $\gamma$ </td><td>Generic sufficiently regular path, possibly with auxiliary randomness  $\xi .$ </td></tr><tr><td> $\gamma _ { \psi }$ </td><td>Unregularized learned residual path.</td></tr><tr><td> $\hat { \gamma } _ { \psi }$ </td><td>Stochastic-regularized learned residual path.</td></tr><tr><td> $p _ { \gamma , t }$ </td><td>Marginal distribution of a generic path  $\gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi )$ </td></tr><tr><td> $p _ { \psi , t }$ </td><td>at time Marginal distribution of the unregularized learned path  $\gamma _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y )$ </td></tr><tr><td> $\widehat { p } _ { \psi , t }$ </td><td>Marginal distribution of the stochastic-regularized path  $\hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon )$ </td></tr><tr><td> $p _ { \phi , t }$ </td><td>ODE marginal induced by the learned sampler  $v _ { \phi } ;$  this is not a path marginal unless  $v _ { \phi }$  equals the corresponding ideal path velocity.</td></tr><tr><td> $v _ { \gamma } ^ { \star }$ </td><td>Population flow-matching target for a generic path  $\gamma ;$  it is the condi- tional mean of the path tangent given position, time, and label.</td></tr><tr><td> $v _ { \psi } ^ { \star }$ </td><td>Population flow-matching target for the unregularized learned path</td></tr><tr><td> $\hat { v } _ { \psi } ^ { \star }$ </td><td>Population flow-matching target for the stochastic-regularized path</td></tr><tr><td> $v _ { \phi }$ </td><td>Trained flow network used for sampling at inference time.</td></tr><tr><td> $u _ { t }$ </td><td>Generic velocity field used in entropy-rate identities.</td></tr><tr><td> $\rho _ { t }$ </td><td>Time-dependent source-perturbation schedule.</td></tr><tr><td> $\sigma _ { t }$ </td><td>Gaussian smoothing scale induced by source perturbation,  $\sigma _ { t } = ( 1 -$   $t ) \sqrt { 1 - \rho _ { t } ^ { 2 } }$ </td></tr><tr><td> $L _ { \mathrm { e u c } } ^ { \rho , \psi } ( t , y )$ </td><td>Pointwise kinetic energy of the stochastic-regularized path, defined in (28).</td></tr></table>

## B Discussion and Limitations

Our experiments focus on the image generation task using FID as the key metric. While this is a challenging problem, flow-based generative models have been applied in a broad range of settings. It would be useful to investigate the efficacy of our method in these applications as well.

Path–flow alignment requires additional training compute and memory for the path network, and alternating path and flow optimization adds engineering complexity. The compute-matched results in Appendix F.1 show that the gains persist under comparable continuation compute, but do not remove this practical overhead. Our experiments study continued training from pretrained or converged checkpoints rather than joint training from scratch. Although Appendix F.3 broadens the architecture and dataset scope, the evaluation remains restricted to class-conditional image generation; text-toimage models and non-image domains remain untested. The sensitivity study over $\rho _ { t }$ and $\beta$ is limited, the path network and subtractor design is not exhaustively ablated, and the multi-seed results in Appendix F.2 measure generation-seed variability rather than variability across independently trained models.

Theorem 1 and Proposition 3 control the stochastic training-path marginals and their path-induced population velocity, not directly the marginals produced by the finite learned sampler. These marginals coincide when the learned flow equals the path-induced population velocity, but may differ under finite optimization and model capacity; velocity-level flow-matching control alone does not provide the derivative-level control needed to transfer the entropy guarantee. Accordingly, the observed association between late-time divergence, entropy bottlenecks, and FID is empirical rather than causal. Finally, the Gaussian-convolution, entropy, and Fisher-information guarantees require the stated conditional-independence condition and do not automatically extend to general endpoint couplings such as optimal transport.

## C Path network and subtractor implementation

Below we describe the implementation of the residual path network $R _ { \psi }$ used in the SiT experiments. The final sampler is still an ordinary SiT velocity model $v _ { \phi }$ . The additional path machinery consists of two SiT-shaped networks used only to define the training path: a trainable path network $P _ { \psi }$ and a frozen subtractor network S. The role of the subtractor is to make the learned path start exactly as the linear path while still allowing $P _ { \psi }$ to learn endpoint-conditioned residual curvature.

Teacher-residualized path parameterization. Let

$$
\gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) = ( 1 - t ) x _ { 0 } + t x _ { 1 } .
$$

The implemented path is

$$
\begin{array} { r l } & { \gamma _ { \psi } ^ { \mathrm { i m p l } } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon _ { 0 } ) = \gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) } \\ & { \qquad + t ( 1 - t ) \Big [ P _ { \psi } ( \hat { \gamma } _ { \mathrm { l i n } } ( t ) , \hat { \Delta } ( t ) , t , y ) - S ( \hat { \gamma } _ { \mathrm { l i n } } ( t ) , t , y ) \Big ] . } \end{array}\tag{31}
$$

thus (31) matches the residual form in (6) with

$$
\begin{array} { r } { R _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) = P _ { \psi } ( \gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) , x _ { 1 } - x _ { 0 } , t , y ) - S ( \gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } ) , t , y ) . } \end{array}
$$

Perturbed endpoint inputs. The outer anchor path in (31) always uses the true endpoints $( x _ { 0 } , x _ { 1 } )$ Perturbations are applied only to the inputs of $P _ { \psi }$ and S. Given schedules $\rho _ { t } \in [ 0 , 1 ]$ , define

$$
\hat { x } _ { 0 } ( t ) = \rho _ { t } x _ { 0 } + \sqrt { 1 - \rho _ { t } ^ { 2 } } \epsilon _ { 0 } .\tag{32}
$$

The source perturbation $\rho _ { t }$ is the stochastic regularizer analyzed in Section 4.2. The actual inputs to the path network and subtractor are

$$
\begin{array} { r } { \hat { \gamma } _ { \mathrm { l i n } } ( t ) = ( 1 - t ) \hat { x } _ { 0 } ( t ) + t x _ { 1 } , \qquad \hat { \Delta } ( t ) = x _ { 1 } - \hat { x } _ { 0 } ( t ) . } \end{array}\tag{33}
$$

Thus stochastic regularization hides part of the endpoint information from the residual branch, while the endpoint identities of the learned path remain exact because the residual term is multiplied by the boundary envelope.

Subtractor network. The subtractor $S ( x , t , y )$ is an unmodified SiT velocity network with the standard single patch-embedding stem, timestep embedding, class-label embedding, SiT transformer blocks, and final velocity head. In the main experiments, S is loaded from a pretrained SiT-S/2 checkpoint and frozen. During subsequent flow training, both the learned path network and the subtractor are frozen; together they define the training states and tangent targets for the flow model $v _ { \phi }$

Dual-stem path network. The path network $P _ { \psi }$ uses the same SiT transformer body as $S ,$ but replaces the single spatial input stem with two stems. The first stem embeds the hatted linear state and the second stem embeds the hatted endpoint displacement:

$$
z _ { \mathrm { l i n } } = \mathrm { E m b e d } _ { \mathrm { l i n } } ( \hat { \gamma } _ { \mathrm { l i n } } ( t ) ) , \qquad z _ { \Delta } = \mathrm { E m b e d } _ { \Delta } ( \hat { \Delta } ( t ) ) .
$$

The token sequence entering the transformer is

$$
z = z _ { \mathrm { l i n } } + z _ { \Delta } + z _ { \mathrm { p o s } } ,\tag{34}
$$

where $z _ { \mathrm { p o s } }$ is the fixed sinusoidal positional embedding used by SiT. The usual SiT conditioning vector is

$$
c = e _ { t } ( t ) + e _ { y } ( y ) ,
$$

where $e _ { t }$ is the timestep embedding and $e _ { y }$ is the class-label embedding. When endpoint conditioning is enabled, the delta-stem tokens are also average pooled and passed through a small projector,

$$
c = e _ { t } ( t ) + e _ { y } ( y ) + W _ { \mathrm { e n d } } \mathrm { S i L U } \left( \operatorname { L a y e r N o r m } \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } z _ { \Delta , i } \right) \right) .\tag{35}
$$

This global endpoint-conditioning branch is the EndpointConditionProjector module, and the full path network is implemented as DualStemPathSiT. The transformer blocks and final head are otherwise the same as in the base SiT velocity model.

Initialization and exact linear startup. The path network $P _ { \psi }$ and subtractor $S$ are initialized from the same pretrained SiT checkpoint. Loading the checkpoint into $P _ { \psi }$ proceeds as follows. The vanilla SiT input stem is copied into the linear stem $\mathrm { E m b e d _ { l i n } }$ . The timestep embedder, label embedder, positional embedding, transformer blocks, and final layer are copied unchanged. The new delta stem Embed $\Delta$ is zero-initialized, and the final linear layer of the endpoint-conditioning projector is also zero-initialized. Consequently, at initialization,

$$
P _ { \psi } ( \hat { \gamma } _ { \mathrm { l i n } } ( t ) , \hat { \Delta } ( t ) , t , y ) = S ( \hat { \gamma } _ { \mathrm { l i n } } ( t ) , t , y ) ,\tag{36}
$$

provided both networks are loaded from the same checkpoint. Therefore the bracketed residual in (31) is exactly zero and

$$
\gamma _ { \psi } ^ { \mathrm { i m p l } } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon _ { 0 } ) = \gamma _ { \mathrm { l i n } } ( t ; x _ { 0 } , x _ { 1 } )
$$

at the start of path training. This exact startup property is why we use the path network and subtractor pair rather than directly learning an unconstrained displacement from scratch.

## D Diagnostic correlations for path overfitting

In Figures 2–3 and 6–8, we use the same settings as SiT-S in Table 1 other than changes to $\beta$ and $\rho$ noted in each figure’s legends. We provide more detailed descriptions for Figures 2–3: curves with $\rho _ { t } = 1$ remove stochastic source regularization and vary kinetic regularization through $\beta ,$ , while $\rho _ { t } = 0 . 5$ and $\rho _ { t } = 1 - t$ are stochastic variants at $\beta = 0 . 0 { \dot { 5 } }$ . Stronger kinetic regularization alone reduces (but does not prevent) late degradation.

This appendix provides the scatter-plot analysis behind the diagnostic discussion in Sections 3.5 and 4.1. All points are SiT-S runs evaluated with 50k FID. Each plot contains checkpoints after 4k, 8k, and 12k flow-training steps, with marker shape indicating the step and color indicating the algorithmic variant. The variants sweep the kinetic energy regularization strength $\beta ,$ the source-perturbation schedule $\rho _ { t }$ , and an additional acceleration-regularized ablation. The acceleration-regularized run adds

$$
\lambda \mathbb { E } \int _ { 0 } ^ { 1 } \left\| \partial _ { t } ^ { 2 } \gamma _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y ) \right\| _ { 2 } ^ { 2 } d t
$$

to the path objective, with $\lambda = 2 \times 1 0 ^ { - 5 }$ . Larger values of λ performed poorly in our preliminary sweeps and are not included here. The vertical axis is truncated to the FID range [8, 11], which is the range of interest; higher FID values correspond to extremely bad path-overfitting runs.

We emphasize these plots as qualitative diagnostics rather than as a statistical correlation study. The number of runs is small and the points are coupled through common checkpoints and training schedules. Nevertheless, the comparison is useful: among the measured quantities, the last-bucket divergence gives the clearest visual correlation with FID.

Figure 6 shows the strongest qualitative relationship. The unregularized or weakly regularized checkpoints that degrade in FID develop a large positive final-bin divergence, consistent with the entropy-bottleneck picture in Section 4.1. In contrast, the stochastic source-regularized checkpoints maintain low FID while avoiding a large positive late-time divergence spike. The accelerationregularized baseline can reduce some geometric quantities, but it does not reliably remove this late positive-divergence behavior.

Taken together, Figures 6–8 highlight that the divergence spike appears to be a stronger predictor of path overfitting, compared to other geometric quantities such as speed and acceleration of the path or the flow. The late-time divergence statistic is more directly tied to the entropy-rate identity of Proposition 2, and in this sweep it is the clearest diagnostic of FID degradation.

## E Proofs for Key Results

## E.1 Proof of Proposition 1

Proof of Proposition 1. We first prove the population regression statement. All expectations are taken with respect to the chosen endpoint coupling conditional on $y ,$ , the auxiliary randomness $\xi ,$ and $t \sim \mathrm { U n i f } [ 0 , 1 ]$ . Let

$$
\mathcal { G } = \sigma ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y )
$$

![](images/e39faafd201e08ca2a61551a1664a0a93a38bcafde3ea68fbd469874f4ca4105.jpg)  
Figure 6: FID versus per-dimensional flow divergence in the last time bucket. The last bucket corresponds to the final bin of the ten-bin estimator in (11). Runs with large positive last-bucket divergence are precisely the high-FID overfit checkpoints in this sweep, while the best checkpoints keep the final-bin divergence negative or only mildly positive.

![](images/799a949c4c55bca2e7aaba2658e5c4960301b5f606794318c370978d2009be2f.jpg)  
(a) Average flow Euclidean norm.

![](images/48ebb7c4bc9db18110a9ea389887dc41b893998648b112b367c8a8de67204f49.jpg)  
(b) Average flow acceleration.  
Figure 7: FID versus global flow-geometry summaries. Unlike the last-bucket divergence in Figure $^ { 6 , }$ average flow speed and acceleration are not monotone indicators of sample quality. Some low-FID stochastic source-regularized checkpoints have comparable or larger average flow speed and acceleration than higher-FID unregularized checkpoints.

be the σ-algebra generated by the point on the path, the time, and the label. For any square-integrable vector field $v _ { \phi }$ , the random variable

$$
v _ { \phi } ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y )
$$

is ${ \mathcal { G } } .$ -measurable. By the tower property,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { \mathrm { f o w } } } ( v _ { \phi } \mid \gamma ) = \mathbb { E } \Big [ \left\| v _ { \phi } \big ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y \big ) - \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \right\| _ { 2 } ^ { 2 } \Big ] } \\ & { \quad \quad \quad \quad = \mathbb { E } \Big [ \mathbb { E } \big [ \left\| v _ { \phi } \big ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y \big ) - \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \right\| _ { 2 } ^ { 2 } \big | \mathcal { G } \big ] \Big ] . } \end{array}\tag{37}
$$

Fix a realization of ${ \mathcal { G } } ,$ say

$$
\gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x , \qquad t = s , \qquad y = \bar { y } .
$$

Conditional on this event, the value of the vector field is a deterministic vector $a = v _ { \phi } ( x , s , \bar { y } )$ . The conditional risk is

$$
{  { \mathbb E } } \Big [ \| a - \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \| _ { 2 } ^ { 2 } \Big | \operatorname { \diamond } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x , t = s , y = \bar { y } \Big ] .
$$

![](images/b272917c7a560e9e32b7dc07dd108effcb8413a266f189d6eafe1c16cfc21cd3.jpg)  
(a) Average path Euclidean norm.

![](images/09b64bc33687d5f6855e06ae686676b3f335f3605e146ce15856f67705bfae59.jpg)  
(b) Average path acceleration.  
Figure 8: FID versus global path-geometry summaries. The path-speed and path-acceleration summaries also do not explain the FID ranking on their own. In particular, good stochastic sourceregularized checkpoints can have larger path speed or acceleration than worse unregularized checkpoints. This supports the view that the key failure mode is distributional—a late entropy-expansion spike—rather than simply excessive path length or curvature.

Let

$$
m ( x , s , \bar { y } ) = \mathbb { E } [ \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \mid \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x , t = s , y = \bar { y } ] .
$$

Then

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \| a - \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \| _ { 2 } ^ { 2 } \Big | \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x , t = s , y = \bar { y } \Big ] } \\ & { \qquad = \| a - m ( x , s , \bar { y } ) \| _ { 2 } ^ { 2 } + \mathbb { E } \Big [ \| \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) - m ( x , s , \bar { y } ) \| _ { 2 } ^ { 2 } \Big | \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x , t = s , y = \bar { y } \Big ] ~ , } \end{array}
$$

because the conditional mean of $\dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) - m ( x , s , \bar { y } )$ is zero under the same conditioning. The second term does not depend on a, while the first term is uniquely minimized by $a = m ( x , s , \bar { y } )$ Therefore the population minimizer is

$$
v _ { \gamma } ^ { \star } ( x , t , y ) = \mathbb { E } [ \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \mid \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x , t , y ]
$$

for $p _ { \gamma , t } ( \cdot \mid y )$ dt-almost every $( x , t )$ and for the relevant labels y. This proves the first part of the proposition. Notice that no endpoint-preservation assumption was used.

The same conditional-expectation argument also gives the projection identity

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { f l o w } } ( v _ { \phi } \mid \gamma ) = \mathbb { E } \Big [ \left\| v _ { \phi } ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y ) - v _ { \gamma } ^ { \star } ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y ) \right\| _ { 2 } ^ { 2 } \Big ] } \\ & { \qquad + \mathbb { E } \Big [ \left\| \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) - v _ { \gamma } ^ { \star } ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y ) \right\| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{38}
$$

Indeed, expand

$$
\begin{array} { r l } & { \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) - v _ { \phi } \big ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y \big ) } \\ & { \qquad = \big ( \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) - v _ { \gamma } ^ { \star } ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y ) \big ) } \\ & { \qquad + \big ( v _ { \gamma } ^ { \star } ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y ) - v _ { \phi } \big ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y \big ) \big ) . } \end{array}
$$

The cross term has expectation zero because the second factor is G-measurable and

$$
\begin{array} { r } { \mathbb { E } \big [ \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) - v _ { \gamma } ^ { \star } ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) , t , y ) \mid \mathcal { G } \big ] = 0 . } \end{array}
$$

This proves (38).

We now prove the ODE transport statement. Fix y and assume the additional hypotheses in the second part of Proposition 1. In particular, the Borel, Lipschitz, and linear-growth conditions imply that $v _ { \gamma } ^ { \star } ( \cdot , \cdot , y )$ generates a unique Carathéodory flow. Let $f \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ be a test function. Differentiating under the expectation gives

$$
\frac { d } { d t } \mathbb { E } [ f ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) ) ~ | ~ y ] = \mathbb { E } [ \langle \nabla _ { x } f ( \gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) ) , \dot { \gamma } ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) \rangle ~ | ~ y ] .
$$

Conditioning on $\gamma ( t ; x _ { 0 } , x _ { 1 } , y , \xi ) = x$ and using the definition of $v _ { \gamma } ^ { \star }$

$$
\frac { d } { d t } \int f ( x ) p _ { \gamma , t } ( x \mid y ) d x = \int \left. \nabla _ { x } f ( x ) , v _ { \gamma } ^ { \star } ( x , t , y ) \right. p _ { \gamma , t } ( x \mid y ) d x .
$$

This is the weak form of the continuity equation associated with the velocity field $v _ { \gamma } ^ { \star }$ . The law of the Carathéodory flow

$$
\frac { d } { d t } z ( t ) = v _ { \gamma } ^ { \star } ( z ( t ) , t , y )
$$

satisfies the same continuity equation. By endpoint preservation and $x _ { 0 } \sim p _ { 0 }$ , the path marginal at $t = 0$ is $p _ { 0 }$ , so the ODE and the path have the same initial law. Under the Lipschitz and linear-growth assumptions of Proposition 1, the associated probability solution is unique. Hence the ODE marginals coincide with the path marginals for every $t \in [ 0 , 1 ]$ . Finally, endpoint preservation and $x _ { 1 } \sim p _ { 1 } ( \cdot \mid y )$ imply that the path marginal at $t = 1$ is $p _ { 1 } ( \cdot \mid \mathbf { \bar { \psi } } _ { y } ) , \mathbf { \bar { s } o } z ( 1 ) \sim p _ { 1 } ( \cdot \mid y )$ 口

## E.2 Proof of Proposition 2

Proof. The differential entropy is

$$
h ( p _ { t } ( \cdot \mid y ) ) = - \int p _ { t } ( x \mid y ) \log p _ { t } ( x \mid y ) d x .
$$

Differentiating and using $\textstyle \int \partial _ { t } p _ { t } ( x \mid y ) d x = 0$ gives

$$
\frac { d } { d t } h ( p _ { t } ( \cdot \mid y ) ) = - \int \partial _ { t } p _ { t } ( x \mid y ) \log p _ { t } ( x \mid y ) d x .
$$

Substituting $\partial _ { t } p _ { t } ( x \mid y ) + \nabla _ { x } \cdot ( p _ { t } ( x \mid y ) u _ { t } ( x , y ) ) = 0$ , yields

$$
\frac { d } { d t } h ( p _ { t } ( { } \cdot { } \mid { } y ) ) = \int { \nabla _ { x } } \cdot ( p _ { t } { u _ { t } } ) \log p _ { t } d x .
$$

Integrating by parts and using the decay assumption,

$$
\frac { d } { d t } h ( p _ { t } ( \cdot \mid y ) ) = - \int p _ { t } ( x \mid y ) u _ { t } ( x , y ) \cdot \nabla _ { x } \log p _ { t } ( x \mid y ) d x = \int p _ { t } ( x \mid y ) \nabla _ { x } \cdot u _ { t } ( x , y ) d x .
$$

This proves (12). Integrating over [a, b] proves (13).

## E.3 Proof of Theorem 1

Proof. For the fixed time $t ,$ the pair $( x _ { 0 } , \hat { x } _ { 0 } ( t ) )$ is jointly Gaussian and satisfies

$$
\operatorname { C o v } ( x _ { 0 } ) = I _ { d } , \qquad \operatorname { C o v } ( { \hat { x } } _ { 0 } ( t ) ) = I _ { d } , \qquad \operatorname { C o v } ( x _ { 0 } , { \hat { x } } _ { 0 } ( t ) ) = \rho _ { t } I _ { d } .
$$

The variable $\eta _ { t }$ in (23) is therefore jointly Gaussian with ${ \hat { x } } _ { 0 } ( t )$ . A direct covariance calculation gives

$$
\mathrm { C o v } ( \eta _ { t } ) = I _ { d } , \qquad \mathrm { C o v } ( \eta _ { t } , \hat { x } _ { 0 } ( t ) ) = 0 .
$$

Hence, conditional on $y , \eta _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$ and $\eta _ { t }$ is independent of ${ \hat { x } } _ { 0 } ( t )$ . Under the independent endpoint coupling, $x _ { 1 }$ is independent of $( x _ { 0 } , \epsilon )$ conditional on $y ,$ so η<sub>t</sub> is conditionally independent of $( \hat { x } _ { 0 } ( t ) , x _ { 1 } )$ given $y .$ Rearranging (23) gives

$$
x _ { 0 } = \rho _ { t } \hat { x } _ { 0 } ( t ) + \sqrt { 1 - \rho _ { t } ^ { 2 } } \eta _ { t } .
$$

Substituting this identity into (18) yields (24). Since the last term $\sigma _ { t } \eta _ { t }$ is Gaussian and independent of the non-Gaussian term (25), the conditional law of their sum is the convolution (22).

The entropy floor follows by conditioning on the non-Gaussian term in (25). After this conditioning, the only remaining randomness in the state is the independent Gaussian $\sigma _ { t } \eta _ { t }$ , whose entropy is

$$
\frac { d } { 2 } \log ( 2 \pi e \sigma _ { t } ^ { 2 } ) .
$$

Removing the conditioning can only increase entropy by a nonnegative mutual-information term, giving (26).

It remains to explain the Fisher-information bound. Let $\varphi _ { \sigma _ { t } }$ denote the density of $\mathcal { N } ( 0 , \sigma _ { t } ^ { 2 } I _ { d } )$ . By the convolution representation,

$$
\hat { p } _ { \psi , t } ( x \mid y ) = \int \varphi _ { \sigma _ { t } } ( x - r ) d q _ { \psi , t } ( r \mid y ) .
$$

Differentiating under the integral gives

$$
\nabla _ { x } \hat { p } _ { \psi , t } ( x \mid y ) = - \frac { 1 } { \sigma _ { t } ^ { 2 } } \int ( x - r ) \varphi _ { \sigma _ { t } } ( x - r ) d q _ { \psi , t } ( r \mid y ) .
$$

Equivalently, using the representation $\hat { \gamma } _ { \psi } ( t ) = r + \sigma _ { t } \eta _ { t }$ from (24),

$$
\nabla _ { x } \log \hat { p } _ { \psi , t } ( x \mid y ) = - \frac { 1 } { \sigma _ { t } } \mathbb { E } \left[ \eta _ { t } \mid \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = x , y \right] .\tag{39}
$$

Therefore Jensen’s inequality gives

$$
\begin{array} { r l } & { I ( \hat { p } _ { \psi , t } ( \cdot \mid y ) ) = \displaystyle \frac { 1 } { \sigma _ { t } ^ { 2 } } \mathbb { E } \left[ \| \mathbb { E } [ \eta _ { t } \mid \hat { \gamma } _ { \psi } ( t ) , y ] \| _ { 2 } ^ { 2 } \mid y \right] } \\ & { \quad \le \displaystyle \frac { 1 } { \sigma _ { t } ^ { 2 } } \mathbb { E } [ \| \eta _ { t } \| _ { 2 } ^ { 2 } ] = \displaystyle \frac { d } { \sigma _ { t } ^ { 2 } } . } \end{array}
$$

This proves (27).

## E.4 Proof of Proposition 3

Proof. By the definition of the stochastic-path population target in (20), the ideal marginal velocity is

$$
\begin{array} { r } { \hat { v } _ { \psi } ^ { \star } ( x , t , y ) = \mathbb { E } \left[ \partial _ { t } \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) ~ | ~ \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) = x , t , y \right] . } \end{array}
$$

Applying Proposition 2 to the probability flow generated by $\hat { v } _ { \psi } ^ { \star }$ gives

$$
\frac { d } { d t } h ( \hat { p } _ { \psi , t } ( \cdot \mid y ) ) = \mathbb { E } _ { x \sim \hat { p } _ { \psi , t } ( \cdot \mid y ) } [ \nabla _ { x } \cdot \hat { v } _ { \psi } ^ { \star } ( x , t , y ) ] .
$$

Integrating by parts,

$$
\begin{array} { r } { \frac { d } { d t } h ( \hat { p } _ { \psi , t } ( \cdot \mid y ) ) = - \mathbb { E } _ { x \sim \hat { p } _ { \psi , t } ( \cdot \mid y ) } \left[ \left. \hat { v } _ { \psi } ^ { \star } ( x , t , y ) , \nabla _ { x } \log \hat { p } _ { \psi , t } ( x \mid y ) \right. \right] } \\ { \geq - \sqrt { \mathbb { E } _ { x \sim \hat { p } _ { \psi , t } ( \cdot \mid y ) } \left\| \hat { v } _ { \psi } ^ { \star } ( x , t , y ) \right\| _ { 2 } ^ { 2 } } \sqrt { I ( \hat { p } _ { \psi , t } ( \cdot \mid y ) ) } , } \end{array}
$$

where the inequality is Cauchy–Schwarz.

The first factor is controlled by the path kinetic energy. Indeed, by Jensen’s inequality for conditional expectation,

$$
\begin{array} { r l } & { \mathbb { E } _ { x \sim \hat { p } _ { \psi , t } ( \cdot | y ) } \left\| \hat { v } _ { \psi } ^ { \star } ( x , t , y ) \right\| _ { 2 } ^ { 2 } \leq \mathbb { E } \left[ \left\| \partial _ { t } \hat { \gamma } _ { \psi } ( t ; x _ { 0 } , x _ { 1 } , y , \epsilon ) \right\| _ { 2 } ^ { 2 } \middle | y \right] } \\ & { \qquad = L _ { \mathrm { e u c } } ^ { \rho , \psi } ( t , y ) . } \end{array}
$$

The second factor is controlled by Gaussian smoothing: Theorem 1 gives

$$
I ( \hat { p } _ { \psi , t } ( \cdot \mid y ) ) \leq \frac { d } { ( 1 - t ) ^ { 2 } ( 1 - \rho _ { t } ^ { 2 } ) } .
$$

Combining the last three displays proves (29).

## F Experimental details

We implement Algorithm 1 with flow $v _ { \phi }$ initialized from already-trained SiT [13] checkpoints of the appropriate size and path $( P _ { \psi }$ and $S )$ initialized from pretrained checkpoints. Our main experiments use ImageNet-256 × 256 [5]. Our setting is post-training of a pretrained flow model. For SiT-S/B/L, we finetune the pretrained checkpoints until FID plateaus and use the resulting checkpoint to initialize both the standard continuation baseline and our method. For SiT-XL, finetuning does not improve the pretrained checkpoint FID, so the pretrained checkpoint is the initialization for the flow network $v _ { \phi }$

Our path network $R _ { \psi }$ is constructed from a pre-trained flow network, as described in Appendix C. For the FM experiments in Table 1, we use the pretrained SiT-S checkpoint to initialize $P _ { \psi }$ and S. For the MG experiments in Table 1, we use the pretrained SiT-B-MG checkpoint to initialize $P _ { \psi }$ and S.

Experimental details for FM experiments in Table 1. In each cycle we finetune path $R _ { \psi }$ using maximum learning rate $3 \times 1 0 ^ { - 4 }$ with cosine decay schedule for 3000 steps, batch size 280, $\beta = 0 . 0 5$ and $\rho _ { t } = c \cdot ( 1 - t )$ where $c = 1$ for SiT-S,B,L and $c = 0 . 7$ for SiT-XL. In each cycle we finetune flow $v _ { \phi }$ using constant learning rate $5 \times 1 0 ^ { - 6 }$ for 4000 steps with no EMA, batch size 768, and inherit the same $\rho _ { t }$ from the path tuning stage. To ensure a fair comparison, we apply standard finetuning to the pretrained baseline SiT models using both the default pretraining settings reported in SiT [13] and our settings for finetuning the flow network $v _ { \phi } ,$ , and report the best FID for the baseline given a fixed finetuning budget. Applying standard finetuning to SiT-S,B,L gives improvements to the FID, whereas finetuning SiT-XL results in worse FID. Experiments are conducted on H200 GPUs. We compute FID [6] and Inception-Score (IS) [16] on 50k images for each checkpoint using the Heun ODE sampler at 250 NFE. We tune the classifier-free guidance strength (CFG) [7] separately for the SiT baseline and our method, generally noticing higher CFG compared to the baseline working better for Algorithm 1. We use $\mathrm { C \bar { F } \bar { G } = \{ 3 , 2 , 1 . 5 , \bar { 1 . 5 } \} }$ for baseline SiT-S,B,L,XL models, and CFG $= \{ 3 . 5 , 2 . 5 , 2 . 0 , 1 . 7 \}$ for our $\operatorname { S i T - S } , \mathbf { B } , \mathbf { L } , \operatorname { X I }$ models. To ensure a fair comparison, we sweep CFG on the same grid for both the SiT baseline and our method, and report FID with best CFG.

Experimental details for MG experiments in Table 1. For the model-guidance experiments, Algorithm 1 uses three alternating cycles, each with 3k path-tuning steps followed by 4k MG-flowtuning steps. Both SiT-B-MG and SiT-XL-MG use $\beta = 0 . 0 5$ , flow EMA decay 0.999, and MG-flow tuning with batch size 768 and learning rate $5 \times 1 0 ^ { - 6 }$ . For both models, the path network $P _ { \psi }$ and subtractor S are initialized from the pretrained SiT-B-MG checkpoint and use batch size 140. The SiT-B-MG run uses a constant source-perturbation $\rho _ { t } = 0 . 5$ and path learning rate $3 \times 1 0 ^ { - 4 }$ , while SiT-XL-MG uses $\rho _ { t } = 0 . 7 ( 1 - t )$ and the path learning rate is $5 \times 1 0 ^ { - 5 }$ . For a fair comparison, the standard MG fine-tuning baselines use the same batch size and learning rate as MG-flow tuning. All entries are evaluated with 50k generated samples using the Heun ODE sampler with 250 NFE and $\mathrm { C F G } = 1 . 0$

Table 3: Full ImageNet- $2 5 6 \times 2 5 6$ results corresponding to Table 1. The baseline columns report the initial checkpoint and the best FID reached by standard fine-tuning within the 12k continuation budget. Our method is evaluated after each alternating cycle.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Params</td><td colspan="2">FID (baseline) ↓</td><td colspan="3">FID (ours) ↓</td><td rowspan="2">IS (baseline) ↑</td><td colspan="3">IS (ours) ↑</td></tr><tr><td>Init</td><td>Best</td><td>4k</td><td>8k</td><td>12k</td><td>Init 4k</td><td>8k</td><td>12k</td></tr><tr><td>SiT-S</td><td>33M</td><td>9.109</td><td>9.022</td><td>8.375</td><td>8.359</td><td>8.492</td><td>173.6</td><td>188.9</td><td>187.2</td><td>185.4</td></tr><tr><td>SiT-B</td><td>130M</td><td>5.074</td><td>5.057</td><td>4.703</td><td>4.480</td><td>4.415</td><td>199.0</td><td>234.9</td><td>229.0</td><td>225.5</td></tr><tr><td>SiT-L</td><td>458M</td><td>3.424</td><td>3.390</td><td>3.014</td><td>2.882</td><td>2.742</td><td>201.1</td><td>274.8</td><td>271.5</td><td>267.2</td></tr><tr><td>SiT-XL</td><td>675M</td><td>2.091</td><td>2.091</td><td>1.909</td><td>1.845</td><td>1.825</td><td>256.1</td><td>282.9</td><td>285.6</td><td>288.6</td></tr></table>

Table 4: Checkpoint-wise model-guidance FID results corresponding to Table 1. Both methods start from the same model-guidance checkpoint; lower is better.
<table><tr><td>Model</td><td>Method</td><td>0k</td><td>4k</td><td>8k</td><td>12k</td></tr><tr><td rowspan="2">SiT-B-MG (130M)</td><td>Ours</td><td>8.090</td><td>6.704</td><td>5.844</td><td>5.585</td></tr><tr><td>MG baseline</td><td>8.090</td><td>8.011</td><td>7.940</td><td>7.950</td></tr><tr><td rowspan="2">SiT-XL-MG (675M)</td><td>Ours</td><td>1.522</td><td>1.336</td><td>1.322</td><td>1.333</td></tr><tr><td>MG baseline</td><td>1.522</td><td>1.494</td><td>1.504</td><td>1.501</td></tr></table>

Table 5: Training efficiency measured in steps per second per H200 GPU.
<table><tr><td>Model</td><td>Alg 1 (Path Phase)</td><td>Alg 1 (Flow Phase)</td><td>Standard flow matching</td></tr><tr><td>SiT-S</td><td>1.86</td><td>1.39</td><td>4.65</td></tr><tr><td>SiT-B</td><td>1.70</td><td>1.02</td><td>2.08</td></tr><tr><td>SiT-L</td><td>0.85</td><td>0.51</td><td>0.71</td></tr><tr><td>SiT-XL</td><td>0.70</td><td>0.37</td><td>0.50</td></tr></table>

Efficiency. Table 5 compares the efficiency of training flow and path networks in Algorithm 1 to standard flow matching training. The additional cost of using $R _ { \psi }$ during flow training causes our flow training to be less efficient per step. However, the relative cost of $R _ { \psi }$ decreases at larger flow models because $R _ { \psi }$ has constant size across S/B/L/XL. All experiments are conducted on H200 GPUs, and our code supports multi-GPU training and image generation. Appendix F.1 reports complete GPU-hour- and FLOP-matched experiments.

## F.1 Compute-matched baseline continuation

Using our method, three training cycles contain 9k path updates and 12k learned-path flow updates. For SiT-S/B/XL, these cycles cost 1.435/2.546/8.811 EFLOPs, whereas 12k updates of standard continuation cost 0.335/1.272/6.559 EFLOPs. We extend each baseline to the larger of the GPU-hourand FLOP-matched budgets. GPU-hour matching gives the larger budget for all three models sizes.

Table 6: GPU-hour matching for three training cycles. GPU-hours are measured on H200 GPUs.
<table><tr><td>Model</td><td>Ours GPU-h (path + flow)</td><td>Matched baseline steps</td><td>Baseline GPU-h</td></tr><tr><td>SiT-S</td><td> $3 . 5 7 4 \ : ( 1 . 2 1 0 + 2 . 3 6 4 )$ </td><td>61.1k</td><td>3.579</td></tr><tr><td>SiT-B</td><td> $4 . 6 6 7 \ : ( 1 . 4 6 8 + 3 . 1 9 9 )$ </td><td>36.6k</td><td>4.677</td></tr><tr><td>SiT-XL</td><td> $1 2 . 2 1 6 \ : ( 3 . 5 4 6 + 8 . 6 7 0 )$ </td><td>22.8k</td><td>12.243</td></tr></table>

Table 7: Compute-matched ImageNet-256 × 256 continuation. FID is computed from 50k images using the Heun sampler at 250 NFE. The reduction is relative to the stronger standard baseline.
<table><tr><td>Model</td><td>Initialization</td><td>Baseline best</td><td>Compute-matched</td><td>Ours</td><td>Reduction</td></tr><tr><td>SiT-S</td><td>9.109</td><td>9.022</td><td>9.025</td><td>8.359</td><td>7.3%</td></tr><tr><td>SiT-B</td><td>5.074</td><td>5.057</td><td>4.981</td><td>4.415</td><td>11.4%</td></tr><tr><td>SiT-XL</td><td>2.091</td><td>2.091</td><td>2.147</td><td>1.825</td><td>12.7%</td></tr></table>

Longer standard continuation improves only SiT-B beyond its original baseline, while path–flow training gives lower FID for all three model sizes. The inference architecture, NFE, and sampling cost are unchanged. We also evaluated the performance of 12k-step continuation with the original SiT hyperparameters. For SiT-B, our tuned baseline reaches 5.057 FID, compared with 5.533 under the original hyperparameters and 5.074 at initialization. For SiT-XL, the two settings reach 2.134 and 2.095, respectively, compared with 2.091 at initialization.

The cost of path–flow alignment applied to a pretrained checkpoint is small relative to pretraining: SiT-XL/2 pretraining uses an estimated 1275.4 EFLOPs, more than 140× the 8.811 EFLOPs used by three cycles of path–flow training.

## F.2 CFG selection and additional metrics

We select classifier-free guidance (CFG) independently for each baseline and path–flow checkpoint using 10k generated images and the adaptive Dopri5 sampler. Note that headline evaluations use 50k-FID.

Table 8: FID on the 10k-image CFG-selection sweeps. Bold and underline identify the selected value for each checkpoint.
<table><tr><td></td><td colspan="3">SiT-S</td><td></td></tr><tr><td></td><td>CFG 2</td><td>CFG 3</td><td>CFG 3.5</td><td>CFG 4</td></tr><tr><td>Baseline</td><td>18.742</td><td>13.094</td><td>13.099</td><td>13.967</td></tr><tr><td>Ours</td><td>20.364</td><td>11.463</td><td>10.988</td><td>11.439</td></tr><tr><td colspan="5">SiT-L</td></tr><tr><td></td><td>CFG 1</td><td>CFG 1.5</td><td>CFG 2</td><td>CFG 2.5</td></tr><tr><td>Baseline</td><td>17.612</td><td>6.992</td><td>7.745</td><td>10.828</td></tr><tr><td>Ours</td><td>24.455</td><td>8.050</td><td>5.606</td><td>7.666</td></tr></table>

<table><tr><td colspan="5">SiT-B</td></tr><tr><td></td><td>CFG 1</td><td>CFG 2</td><td>CFG 2.5</td><td>CFG 3</td></tr><tr><td>Baseline</td><td>31.481</td><td>8.794</td><td>9.157</td><td>11.126</td></tr><tr><td>Ours</td><td>43.778</td><td>9.576</td><td>7.286</td><td>8.051</td></tr><tr><td colspan="5">SiT-XL</td></tr><tr><td></td><td>CFG 1</td><td>CFG 1.5</td><td>CFG 1.7</td><td>CFG 2</td></tr><tr><td>Baseline</td><td>12.446</td><td>4.977</td><td>5.271</td><td>6.771</td></tr><tr><td>Ours</td><td>14.796</td><td>5.062</td><td>4.787</td><td>5.651</td></tr></table>

The sweeps select CFG 3/2/1.5/1.5 for the SiT-S/B/L/XL baselines and 3.5/2.5/2/1.7 for the corresponding path–flow checkpoints. We use the best CFG settings for each model size and method in our 50k-FID headline results.

In Table 9 we additionally report precision and recall corresponding to the SiT checkpoints used for headline results in Table 1. As baseline, we generate using the model with the best FID at every SiT model size.

Table 9: Precision and recall corresponding to the 50k-image headline FID results in Table 1, generated using Heun at 250 NFE. Bold marks the better value within each model and metric.
<table><tr><td></td><td>SiT-S Precision ↑</td><td>Recall ↑</td><td>SiT-B Precision ↑</td><td>Recall ↑</td></tr><tr><td rowspan="2">Baseline Ours</td><td>0.874</td><td>0.283 0.297</td><td>0.850</td><td>0.441</td></tr><tr><td>0.860</td><td></td><td>0.826</td><td>0.474</td></tr><tr><td rowspan="2"></td><td>SiT-L</td><td></td><td>SiT-XL</td><td></td></tr><tr><td>Precision ↑</td><td>Recall ↑</td><td>Precision ↑</td><td>Recall ↑</td></tr><tr><td rowspan="2">Baseline Ours</td><td>0.819</td><td>0.542</td><td></td><td>0.581</td></tr><tr><td>0.806</td><td>0.564</td><td>0.814 0.791</td><td>0.611</td></tr></table>

Recall increases when comparing our method to the baseline, while precision decreases.

To examine how both metrics vary with guidance, Table 10 reports how precision and recall change with respect to CFG using 10k images. Bold marks the best value in each row, and underline marks the CFG selected by FID for the headline evaluation.

Table 10: Precision and recall across 10k-image CFG sweeps using the Dopri5 sampler.
<table><tr><td colspan="6">SiT-S</td></tr><tr><td>Metric</td><td>Method</td><td>CFG 2</td><td>CFG 3</td><td>CFG 3.5</td><td>CFG 4</td></tr><tr><td>Precision ↑</td><td>Baseline</td><td>0.850</td><td>0.875</td><td>0.872</td><td>0.868</td></tr><tr><td></td><td>Ours</td><td>0.831</td><td>0.858</td><td>0.857</td><td>0.852</td></tr><tr><td>Recall ↑</td><td>Baseline</td><td>0.360</td><td>0.364</td><td>0.354</td><td>0.332</td></tr><tr><td></td><td>Ours</td><td>0.385</td><td>0.433</td><td>0.416</td><td>0.407</td></tr><tr><td colspan="6">SiT-B</td></tr><tr><td>Metric</td><td>Method</td><td>CFG 1</td><td>CFG 2</td><td>CFG 2.5</td><td>CFG 3</td></tr><tr><td>Precision ↑</td><td>Baseline</td><td>0.766</td><td>0.853</td><td>0.848</td><td>0.840</td></tr><tr><td></td><td>Ours</td><td>0.732</td><td>0.811</td><td>0.824</td><td>0.815</td></tr><tr><td>Recall ↑</td><td>Baseline</td><td>0.380</td><td>0.523</td><td>0.525</td><td>0.506</td></tr><tr><td></td><td>Ours</td><td>0.326</td><td>0.559</td><td>0.578</td><td>0.580</td></tr><tr><td colspan="6">SiT-L</td></tr><tr><td>Metric</td><td>Method</td><td>CFG 1</td><td>CFG 1.5</td><td>CFG 2</td><td>CFG 2.5</td></tr><tr><td>Precision ↑</td><td>Baseline</td><td>0.776</td><td>0.816</td><td>0.822</td><td>0.813</td></tr><tr><td></td><td>Ours</td><td>0.745</td><td>0.788</td><td>0.798</td><td>0.794</td></tr><tr><td>Recall ↑</td><td>Baseline</td><td>0.518</td><td>0.606</td><td>0.616</td><td>0.604</td></tr><tr><td></td><td>Ours</td><td>0.484</td><td>0.601</td><td>0.655</td><td>0.663</td></tr><tr><td colspan="6">SiT-XL</td></tr><tr><td>Metric</td><td>Method</td><td>CFG 1</td><td>CFG 1.5</td><td>CFG 1.7</td><td>CFG 2</td></tr><tr><td>Precision ↑</td><td>Baseline</td><td>0.776</td><td>0.811</td><td>0.809</td><td>0.808</td></tr><tr><td></td><td>Ours</td><td>0.752</td><td>0.789</td><td>0.795</td><td>0.790</td></tr><tr><td>Recall ↑</td><td>Baseline</td><td>0.564</td><td>0.653</td><td>0.663</td><td>0.671</td></tr><tr><td></td><td>Ours</td><td>0.576</td><td>0.671</td><td>0.688</td><td>0.693</td></tr></table>

At the CFG selected for each path–flow checkpoint, recall changes from 0.354/0.525/0.616/0.663 to 0.416/0.578/0.655/0.688 for SiT-S/B/L/XL, while precision changes from 0.872/0.848/0.822/0.809 to 0.857/0.824/0.798/0.795. At the lowest tested CFG, path–flow training has lower precision for all four sizes; recall is lower for SiT-B/L and slightly higher for SiT-S/XL.

Multi-seed evaluation. In Table 11 we evaluate each fixed checkpoint on three sets of images generated from three disjoint sets of random seeds. The ranges for the random seeds are 0-49,999, 50k-99,999, 100k-149,999.

The mean FID is lower for path–flow training at every model size. Each gap is at least 5.8× the larger sample standard deviation observed for that model.

Table 11: Mean ± sample standard deviation of 50k-image FID across three generation-seed ranges, using Heun at 250 NFE.
<table><tr><td>Model</td><td>Baseline best</td><td>Ours best</td></tr><tr><td>SiT-S</td><td> $9 . 1 4 2 \pm 0 . 1 3$ </td><td> $\mathbf { 8 . 3 8 3 \pm 0 . 0 4 1 }$ </td></tr><tr><td>SiT-B</td><td> $5 . 0 9 3 \pm 0 . 0 4 1$ </td><td> $\mathbf { 4 . 4 7 0 \pm 0 . 0 5 4 }$ </td></tr><tr><td>SiT-L</td><td> $3 . 4 5 8 \pm 0 . 0 6 1$ </td><td> $\mathbf { 2 . 7 7 7 \pm 0 . 0 3 1 }$ </td></tr><tr><td>SiT-XL</td><td> $2 . 1 0 3 \pm 0 . 0 2 6$ </td><td> $\mathbf { 1 . 8 3 0 \pm 0 . 0 1 5 }$ </td></tr></table>

## F.3 Applying our method in more scenarios

Using our method on FlowDCN. To broaden the applicability of our method, we apply three cycles of path–flow training to a pretrained 120M-parameter FlowDCN-B model [20], which has a purely convolutional backbone. We select CFG using 10k-FID (dopri5 sampler, see Table 12), and compare with 25.9k steps of FLOP-matched standard continuation. Final FID is computed from 50k images using the Heun sampler at 250 NFE.

Table 12: FlowDCN-B 10k-image CFG sweep using the 100-step adaptive Dopri5 sampler.
<table><tr><td>Checkpoint</td><td>CFG 1.5</td><td>CFG 2.0</td><td>CFG 2.5</td></tr><tr><td>Baseline</td><td>11.890</td><td>9.142</td><td>11.230</td></tr><tr><td>Ours</td><td>15.734</td><td>8.204</td><td>8.765</td></tr></table>

The sweep selects CFG 2.0 for both checkpoints. At the chosen CFG, Table 13 compares the performance of the pretrained model, compute-matched continuation, and our method’s three-cycle finetuned checkpoint. Path–flow training reaches 5.473 FID, a 7.0% improvement over the best FLOP-matched continuation result of 5.888, demonstrating that our path–flow alignment method can be applied to various architectures in addition to SiT [13].

Table 13: Performance of path–flow alignment using FlowDCN-B (Ours best) compared to continued finetuning (Baseline best), 50k-FID (Heun sampler 250 NFE).
<table><tr><td>Model</td><td>Params</td><td>Init FID</td><td>Baseline best</td><td>Ours best</td><td>FID gain</td></tr><tr><td>FlowDCN-B</td><td>120M</td><td>6.489</td><td>5.888</td><td>5.473</td><td>7.0%</td></tr></table>

Comparison to [17] on CIFAR-10. We compare with Learning Straight Flows [17] by initializing both our path–flow alignment method and [17]’s method from a SiT-CIFAR/2 checkpoint pretrained for 200k steps. This is a smaller variant of SiT adapted for CIFAR-10 that we pretrained. To ensure fair comparisons, we use matched training FLOPs: our path–flow alignment uses 32.888 PFLOPs, whereas our reproduction of Learning Straight Flows [17] and CIFAR-10 baseline continuation use at least as much FLOPs as was allocated to our method. We compare the methods using 10k-FID at CFG 2.0 and the ODE Euler sampler at 2 and 40 NFE.

Our implementation of Learning Straight Flows [17] includes its empirical target-field estimator, joint flow and straightness objective, and a four-block GLOW/PLU invertible interpolant. We use 100 queries and a balanced 500-image candidate pool. Since we are unable to access the original code and some CIFAR-10 hyperparameters are unavailable, we are unable to reproduce [17] exactly.

The standard-continuation row reports the best FID at each NFE achieved during continued finetuning. We report the FID at the end of training for our path–flow alignment and [17]. Learning Straight Flows gives the best result at 2 NFE, while path–flow alignment gives the best result at 40 NFE. Intuitively, our method learns a more curved trajectory (see Figure 1) whereas [17] penalizes the curvature of the resulting trajectory at the cost of high-NFE generation, leading our method to perform better at higher NFE.

Table 14: FLOP-matched CIFAR-10 comparison using 10k images from one generation seed. Lower FID is better.
<table><tr><td>Method</td><td>FID, 2 NFE↓</td><td>FID, 40 NFE↓</td></tr><tr><td>Our method (1 cycle)</td><td>170.845</td><td>9.097</td></tr><tr><td>Baseline continuation</td><td>100.518 (step 1,000)</td><td>10.339 (step 1,500)</td></tr><tr><td> $[ 1 7 ] , \lambda _ { \operatorname* { m a x } } = 0 . 0 0 1 , { \mathrm { G L O W } } \mathrm { L R } 5 \times 1 0 ^ { - 6 }$ </td><td>79.150</td><td>11.667</td></tr><tr><td> $[ 1 7 ] , \lambda _ { \operatorname* { m a x } } = 0 . 0 0 1 , { \mathrm { G L O W } } \sqcup { \mathrm { R } } 5 \times 1 0 ^ { - 7 }$ </td><td>84.230</td><td>11.568</td></tr><tr><td> $[ 1 7 ] , \lambda _ { \operatorname* { m a x } } = 0 . 0 0 0 3 , { \mathrm { G L O W ~ L R ~ 5 \times 1 0 ^ { - 7 } ~ } }$ </td><td>94.295</td><td>11.878</td></tr></table>

## F.4 Qualitative comparisons with our method

Qualitative samples. Figure 9 compares images generated by the pretrained SiT-XL checkpoint with the cycle-3 checkpoint of our path–flow alignment. The grids are uncurated and matched by class and generation seed.

![](images/acfe010589d53083b4585a0e14e4e68c9de3d0be6ea552db6aba01b4b66c16ec.jpg)  
Figure 9: Uncurated SiT-XL samples matched by class and generation seed. Left: the best baseline, given by the framework initialization. Right: path–flow alignment after three cycles.

Most corresponding samples remain similar in class, composition, pose, and background, indicating that our path–flow alignment method refines the pretrained model rather than substantially changing its overall behavior.

Several samples nevertheless show clearer local structure with our method. At (1,3), the green lizard has more clearly defined feet and contact with the surface. $\operatorname { A t } \left( 2 , 3 \right)$ , the baboon has a more coherent head and body. At (3,1), the white wolf has a better-formed face. At (3,2), our result is recognizably a triumphal arch, whereas the baseline resembles an irregular building with a small opening. At (4,4), the rubber eraser is cleanly separated from the surrounding material rather than bleeding into it.

Visualizing learned trajectories. Figure 1 visualizes the trajectories from a toy experiment under standard flow matching, unregularized path learning, and stochastic path learning. Our method induces a trajectory with higher curvature compared to standard flow matching.

Our toy example transports an independently coupled two-component Gaussian mixture centered at $( - 5 , \pm 1 )$ to one centered at $( 5 , \pm 1 )$ ; each component has isotropic standard deviation 0.3. We first train a standard flow-matching teacher on linear paths, then initialize the $\rho = 1$ and $\rho _ { t } = 1 - t$ branches from this teacher and run two cycles of alternating path and flow updates. The plotted curves integrate the same 600 source samples with Heun over 240 steps, allowing their geometry to be compared directly.