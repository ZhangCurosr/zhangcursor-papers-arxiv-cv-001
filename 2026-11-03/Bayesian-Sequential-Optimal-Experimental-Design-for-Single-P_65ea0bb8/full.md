# Bayesian Sequential Optimal Experimental Design for Single-Pixel Imaging using Reinforcement Learning

Marcos Antonio Obando1,2,\*, Tianyuan Wang1, Tanja Tarvainen2, Tristan van Leeuwen1,3, Felix Lucka1

E-mail: Marcos.Obando@cwi.nl

Keywords: experimental design, single-pixel imaging, reinforcement learning

## Abstract

Adaptive compressed sensing (CS) aims to select the most informative measurements sequentially, which is crucial for many modern imaging systems. This task can be formulated as a Bayesian sequential optimal experimental design (sOED) problem, where each new acquisition is chosen to maximally reduce uncertainty about the unknown image given prior information and subsequent data provided by acquisitions. However, in single-pixel imaging (SPI), the large design space consists of all possible binary illumination patterns. Directly optimizing the Bayesian sOED objective over this space using conventional methods is generally challenging.

In this work, we reformulate sOED for SPI as a reinforcement learning (RL) problem. In RL, an agent learns a policy through repeated experimental trials to maximize a reward, here given as the expected information gain, the most complete measure of uncertainty reduction in Bayesian OED. This avoids the need to directly optimize complex information-theoretic objectives during a measurement to obtain the next design. Instead, the result is a learned policy function that directly maps the current posterior belief about the image to the next optimal binary measurement pattern, in a computationally efficient way. Using a simple yet representative SPI imaging scenario, this work evaluates the scalability of the learned policy under two different policy parameterizations. Furthermore, we investigate how performance deteriorates when only approximate posterior information is available or the measurements are impaired by noise, and propose several strategies to introduce additional design constraints motivated by biomedical applications. The numerical experiments demonstrate that the proposed Bayesian sOED-RL framework remains robust and can be extended to more complex, large-scale SPI scenarios.

## 1 Introduction

Single-pixel imaging (SPI) emerged as an alternative to conventional arraybased imaging systems, with a simplified hardware and operation in wavelengths where detector arrays are expensive or unavailable [1] as their main advantages. Unlike traditional cameras, where an entire scene simultaneously using pixel arrays is captured, here an image is sequentially acquired as follows: A spatial light pattern illuminates the scene and a single photodetector records the total light intensity being reflected back. In practice, the patterns are structured via spatial light modulators or digital micromirror devices, the latter producing binary (on/off) illumination and making binary patterns a natural fit for SPI hardware [2, 3]. A clever choice of sensing patterns and reconstruction routines enable the recovery of images from far fewer measurements than the total number of pixels [4, 5]. Known as compressed sensing (CS), the mathematical framework display which properties should signals have so as to be exactly recovered from undersampled measurements, as long as the sensing patterns satisfy certain conditions.

Early approaches chose patterns for this CS framework in a generic manner, such as random binary projections or scrambled Hadamard matrices [6], which provide reasonable results over a broad range of images. When prior knowledge about the samples that are images is available, one can improve the results by adapting the acquisitions. In an optimal experimental design (OED) framework, a representation of the possible images enables the selection of maximally informative measurement patterns, allowing an efficient image recovery [7].

While conventional OED determines a fixed set of optimal patterns before any measurements are taken, sequential OED (sOED) proposes a way more efficient strategy [8]. Instead of deploying all patterns upfront, sOED adaptively selects each new pattern during the measurement process, based on what has already been learned from previous measurements. As early measurements mainly localize where interesting features lie, later ones focus on details around those specific regions. The 'a-posteriori' distribution, a Bayesian update of the prior knowledge by each data acquisition, results in a natural state representation that captures all available information.

However, sOED for imaging faces a severe computational challenge. On each step, one must solve an optimization problem over all possible binary measurement patterns to find the one that maximally reduces the uncertainty about the image. For an image with M pixels, the 2M possible binary patterns make conventional per-step optimization computationally prohibitive, even at modest image sizes [9, 10], limiting the practical applicability of sOED to real-world imaging problems.

To overcome this barrier, we pose sOED as a reinforcement learning (RL) problem. In this approach, rather than solving an optimization problem on each step, one looks for a parametric policy mapping the current posterior distribution directly to measurement patterns. Using the expected information gain between prior and posterior image knowledge as the reward signal, policy training consists on selecting the most informative patterns given what has already been observed along experiments. At evaluation time, the policy acts as a pattern proposal function, without needing any

further optimization.

In this work, two kinds of policy parameterizations are proposed and their scalability is studied. A categorical policy assigns a probability to each of the $2 ^ { M }$ possible binary patterns, whereas a element-wise Bernoulli policy sets each pixel's activation probability independently. Under a controlled SPI scenario, where the optimal policy is known and information gain can be computed exactly, their performance is compared after training against random and optimal design strategies. Moreover, the replacement of the posterior distribution as a state variable by a sample-based approximate and the addition of noise to the measurements are analyzed, as well as hard and soft energy constraints over the binary projections are proposed.

The paper is organized as follows. Section 2 presents the elements of single-pixel imaging and its treatment as an inverse problem, how Bayesian optimal experimental design can be applied to it and its close relation to sequential decision making processes. Section 3 bridges this theory using a reinforcement learning abstraction, and illustrates the technical implementation details for the single-pixel imaging case. Section 4 introduces setting considerations for the proposed learning scenario and numerical results obtained in the studies, with a discussion in Section 5 and conclusions in Section 6.

## 2 Background

## 2.1 Single-pixel imaging as a Bayesian inverse problem

A single measurement in SPI can be described as

$$
y = \langle \mathbf { h } , \mathbf { x } \rangle + \varepsilon ,\tag{1}
$$

where $\mathbf { x } \in \mathbb { R } ^ { M }$ is the image that one wants to recover in vector form, h $N \in$ $\{ 0 , 1 \} ^ { M }$ is a discrete binary pattern and $\varepsilon$ is a noise realization, usually modeled as a normally distributed uncorrelated random variable ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ with mean zero and standard deviation $\sigma .$ The SPI forward model, collecting N observations $\mathbf { y } _ { N } \in \mathbb { R } ^ { N }$ , is described in a matrix-vector form as

$$
\begin{array} { r } { { \bf y } _ { N } = { \bf H } _ { N } { \bf x } + \varepsilon _ { N } \quad \quad } \\ { \left[ y _ { 1 } \right] = \left[ { \bf h } _ { 1 } ^ { T } \right] \left[ \begin{array} { c } { x _ { 1 } } \\ { \vdots } \\ { x _ { M } } \end{array} \right] + \left[ \begin{array} { c } { \varepsilon _ { 1 } } \\ { \vdots } \\ { \varepsilon _ { N } } \end{array} \right] \quad , \quad \quad } \end{array}\tag{2}
$$

with ${ \bf H } _ { N }$ being the forward operator, which rows consist of discrete binary patterns $\mathbf { h } _ { n }$ . Assuming that the noise from individual measurements is uncorrelated, the noise variable $\varepsilon _ { N }$ follows a multivariate normal distribution ${ \boldsymbol \varepsilon } _ { N } \sim \mathcal { N } ( \mathbf { 0 } _ { N } , \sigma ^ { 2 } \mathbf { I } _ { N } )$ When $N \ < \ M$ , the inverse problem of recovering x from measurements $\mathbf { y } _ { N }$ is underdetermined, and usually ill-conditioned, which means that a perturbation in the image x does not necessarily lead to an identifiable change in the observed data $\mathbf { y } _ { N }$ [11].

To prevent this, assumptions based on 'a-priori' knowledge of the image can be used to regularize the possible solutions of the problem. In combination with how consistent measurements are with respect to the forward model, an updated, 'a-posteriori' knowledge of the image is obtained. This methodology can be formalized using the Bayesian framework, where x is treated as a random variable rather than merely a real-valued vector, describing the solution to Eq. (2) through the Bayes formula [12]

$$
P ( \mathbf { x } \mid \mathbf { y } _ { N } ; \mathbf { H } _ { N } ) = { \frac { P ( \mathbf { y } _ { N } \mid \mathbf { x } ; \mathbf { H } _ { N } ) P ( \mathbf { x } ) } { P ( \mathbf { y } _ { N } ; \mathbf { H } _ { N } ) } }\tag{3}
$$

Here, the prior distribution $P ( \mathbf { x } )$ encompasses the knowledge of the image before measurement takes place and can encode the properties one expects the image to have, e.g., total variation priors were proposed to model images with clearly defined edges between piecewise constant areas [13]. Note that the prior distribution is independent of the measurement operator ${ \bf { H } } _ { N }$ . The likelihood distribution $P ( \mathbf { y } _ { N } \mid \mathbf { x } ; \mathbf { H } _ { N } )$ quantifies the probability of measuring $\mathbf { y } _ { N }$ , which contains all measurements up to step N, given the image x using the measurement operator ${ \bf H } _ { N }$ . Here the noise and the unknown x are mutually independent variables

$$
\begin{array} { l } { { \displaystyle P ( { \bf y } _ { N } \mid { \bf x } ; { \bf H } _ { N } ) \propto \exp \left( - \frac { 1 } { 2 { \sigma } ^ { 2 } } \| { \bf H } _ { N } { \bf x } - { \bf y } _ { N } \| _ { 2 } ^ { 2 } \right) } \ ~ } \\ { { \displaystyle ~ = \prod _ { n = 1 } ^ { N } \exp \left( - \frac { ( { \bf h } _ { n } ^ { T } { \bf x } - y _ { n } ) ^ { 2 } } { 2 { \sigma } ^ { 2 } } \right) = : \prod _ { n = 1 } ^ { N } P ( y _ { n } | { \bf x } ; { \bf h } _ { n } ) } } \end{array}\tag{4}
$$

The marginal likelihood $P ( \mathbf { y } _ { N } ; \mathbf { H } _ { N } )$ , also known as model evidence, is a normalizing factor for all possible images x given a particular ${ \bf { H } } _ { N }$

$$
P ( \mathbf { y } _ { N } ; \mathbf { H } _ { N } ) = \mathbb { E } _ { P ( \mathbf { x } ) } \big [ P ( \mathbf { y } _ { N } | \mathbf { x } ; \mathbf { H } _ { N } ) \big ]\tag{5}
$$

As a result, the posterior distribution, $P ( \mathbf { x } \mid \mathbf { y } _ { N } ; \mathbf { H } _ { N } )$ , combines information from the prior and likelihood distributions, assigning a probability score to each possible image x given observed data $\mathbf { y } _ { N }$ . Note that due to the sequential, independent SPI measurement process, the posterior distribution is updated during acquisition as

$$
\begin{array} { r l } { P ( \mathbf { x } | \mathbf { y } _ { N } ; \mathbf { H } _ { N } ) \propto P ( \mathbf { y } _ { N } | \mathbf { x } ; \mathbf { H } _ { N } ) P ( \mathbf { x } ) } & { } \\ { \stackrel { ( 4 ) } { \propto } P ( y _ { N } | \mathbf { x } ; \mathbf { h } _ { N } ) P ( \mathbf { y } _ { N - 1 } | \mathbf { x } ; \mathbf { H } _ { N - 1 } ) P ( \mathbf { x } ) } & { } \\ { \propto P ( y _ { N } | \mathbf { x } ; \mathbf { h } _ { N } ) P ( \mathbf { x } | \mathbf { y } _ { N - 1 } ; \mathbf { H } _ { N - 1 } ) } \end{array}\tag{6}
$$

## 2.2 Bayesian optimal experimental design

The aim in compressed SPI is to obtain a useful reconstruction of $\mathbf { x }$ using $N < M$ measurements. An important question is which of the $2 ^ { N M }$ possible experimental designs $\mathbf { H } _ { N } \in \lbrace 0 , 1 \rbrace ^ { N \times M }$ one should choose for a given prior $P ( \mathbf { x } )$ . Bayesian OED formalizes this problem: First, it uses the Shannon entropy as an information-theoretic measure to quantify the uncertainty about the random image x before the measurement

$$
E n t ( \mathbf { x } ) : = - \mathbb { E } _ { P ( \mathbf { x } ) } \Big [ \log P ( \mathbf { x } ) \Big ]\tag{7}
$$

Then, it assesses how informative a particular collection of measurements $\mathbf { y } _ { N }$ acquired with a design ${ \bf H } _ { N }$ is by introducing the information gain $\mathrm { I G } _ { \mathbf { x } } ( \mathbf { H } _ { N } , \mathbf { y } _ { N } )$ as a difference between prior and posterior entropy

$$
\operatorname { I G } _ { \mathbf { x } } ( \mathbf { y } _ { N } ; \mathbf { H } _ { N } ) : = E n t ( \mathbf { x } ) - E n t ( \mathbf { x } \mid \mathbf { y } _ { N } ; \mathbf { H } _ { N } )\tag{8}
$$

Using the marginal likelihood in (5) and the information gain in (8), the expected information gain $\operatorname { E I G } _ { \mathbf { x } } ( \mathbf { H } )$ provides an average uncertainty change with respect to all possible measurement realizations $\mathbf { y } _ { N }$

$$
\mathrm { E I G } _ { \mathbf { x } } ( \mathbf { H } _ { N } ) : = \mathbb { E } _ { P ( \mathbf { y } _ { N } ; \mathbf { H } _ { N } ) } \big [ \mathrm { I G } _ { \mathbf { x } } ( \mathbf { y } _ { N } ; \mathbf { H } _ { N } ) \big ]\tag{9}
$$

BOED chooses the measurement patterns that maximize the expected information gain

$$
\mathbf H _ { N } ^ { \mathrm { P r } } : = \operatorname * { a r g m a x } _ { \mathbf H _ { N } \in \{ 0 , 1 \} ^ { N \times M } } \mathrm { E I G } _ { \mathbf x } ( \mathbf H _ { N } ) ,\tag{10}
$$

where $\mathbf { H } _ { N } ^ { \operatorname { P r } }$ refers to an 'a-priori’ design , as this choice only relies on information available before the measurement, i.e., the prior. Now consider the case where one can design pattern sequentially during the measurement, based on the information gathered so far, i.e., 'a-posteriori' (sequential Optimal Experimental Design, sOED). In Bayesian sOED, consider the expected information gain of using pattern $\mathbf { h } _ { n }$ for the $n ^ { t h }$ measurement step given that the previous $\mathbf y _ { n - 1 }$ measurements were acquired with the design ${ \bf H } _ { n - 1 }$

$$
\mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } ) : = \mathbb { E } _ { P ( y _ { n } ; \mathbf { h } _ { n } ) } \big [ \mathrm { I G } _ { \mathbf { x } | \mathbf { y _ { n - 1 } } ; \mathbf { H _ { n - 1 } } } ( y _ { n } ; \mathbf { h } _ { n } ) \big ] ,\tag{11}
$$

where the marginal likelihood can be expressed as an expectation over the new measurement $y _ { n }$ likelihood using the the posterior until step $n - 1$

$$
P ( y _ { n } ; \mathbf { h } _ { n } ) : = \mathbb { E } _ { P ( \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } ) } [ P ( y _ { n } | \mathbf { x } ; \mathbf { h } _ { n } ) ] , \quad P ( \mathbf { x } | \mathbf { y } _ { 0 } ; \mathbf { H } _ { 0 } ) : = P ( \mathbf { x } )\tag{12}
$$

Note that the prior $P ( \mathbf { x } )$ in (8), (5) was merely exchanged for the posterior $P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } )$ in (11), (12) and only consider the design of the following pattern $\mathbf { h } _ { n }$ . The greedy sOED $\mathbf { H } _ { N } ^ { \mathrm { G r } } ( \mathbf { x } )$ , which compared to (10) depends on the unknown realization x, maximizes (11) in every step

$$
\mathbf { h } _ { n } ^ { \mathrm { G r } } ( \mathbf { x } ) : = \operatorname * { a r g m a x } _ { \mathbf { h } _ { n } \in \{ 0 , 1 \} ^ { M } } \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { \mathbf { n } - 1 } ; \mathbf { H } _ { \mathbf { n } - 1 } } ( \mathbf { h } _ { n } ) , \quad \mathbf { H } _ { N } ^ { \mathrm { G r } } = [ \mathbf { h } _ { 1 } ^ { \mathrm { G r } } , \dots , \mathbf { h } _ { N } ^ { \mathrm { G r } } ] ^ { T }\tag{13}
$$

Greedy designs reduce the chain of sequential design decisions that all influence the informativeness of the final N-step design ${ \bf { H } } _ { N }$ into a sequence of isolated design decision steps. Although this is often computationally attractive, it is a suboptimal solution to the problem of maximizing the sum of expected information gains over all N steps [14] [15]. As an alternative, the full sequential OED ${ \bf H } _ { N } ^ { \mathrm { F } } ( { \bf x } )$ is the globally optimal sequence of measurement patterns $\mathbf { h } _ { 1 } , \ldots , \mathbf { h } _ { N }$ that maximizes the cumulative expected information gain across all N steps

$$
\mathbf { H } _ { N } ^ { \mathrm { F } } ( \mathbf { x } ) : = \operatorname* { a r g m a x } _ { \mathbf { h } _ { 1 } , \ldots , \mathbf { h } _ { N } \in \{ 0 , 1 \} ^ { M } } \sum _ { n = 1 } ^ { N } \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } )\tag{14}
$$

Unlike greedy approaches that optimize each measurement independently, this formulation accounts for the long-term information-gathering strategy by jointly optimizing over the entire measurement sequence. The key insight is that early measurement choices influence the posterior distributions encountered in subsequent steps, creating a complex interdependence between decisions that greedy methods fail to capture.

The theoretical superiority of the full sOED strategy with respect to the greedy approach, introduced in [8], is formalized as

$$
\sum _ { n = 1 } ^ { N } \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } ^ { \mathrm { F } } } ( \mathbf { h } _ { n } ^ { \mathrm { F } } ) \geq \sum _ { n = 1 } ^ { N } \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } ^ { \mathrm { G r } } } ( \mathbf { h } _ { n } ^ { \mathrm { G r } } )\tag{15}
$$

$$
\sum _ { n = 1 } ^ { N } \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } ^ { \mathrm { F } } } ( \mathbf { h } _ { n } ^ { \mathrm { F } } ) \geq \mathrm { E I G } _ { \mathbf { x } } ( \mathbf { H } _ { N } ^ { \mathrm { P r } } )\tag{16}
$$

Inequality (15) states that the cumulative information gain achieved by the full sOED sequence $\mathbf { H } _ { N } ^ { \mathrm { F } }$ is at least as large as that obtained by the greedy approach $\mathbf { H } _ { N } ^ { \mathrm { G r } }$ , which myopically maximizes the EIG at each individual step without considering future measurements. This gap arises because the full sOED can strategically select measurements that may have lower immediate information gain but enable more informative measurements in subsequent steps a form of temporal credit assignment that greedy methods cannot perform. Inequality (16) further shows that the full sOED outperforms the a-priori design $\mathbf { H } _ { N } ^ { \operatorname { P r } }$ , which selects all N patterns before observing any data and thus cannot adapt to the information revealed by early measurements. Together, these inequalities establish the full sOED as the optimal approach within the Bayesian framework, motivating the development of tractable approximation methods such as the RL-based approach presented in this paper.

Computational considerations. While the full sOED (14) is theoretically superior to the a-priori OED design (10), it comes with similar computational challenges: Taking the interdependence of the terms of the sum in (14) into account means that one also needs to optimize over the exponentially large space $\{ 0 , 1 \} ^ { N M }$ of binary patterns. Even for modest image sizes $( \mathrm { e . g . } , M = 6 4 )$ , the design space contains $2 ^ { 6 4 } \approx 1 0 ^ { 1 9 }$ possible patterns at each step, making exhaustive search or traditional optimization methods computationally prohibitive.

Moreover, computing the expected information gain $\mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } )$ in (11) requires integrating over all possible measurement outcomes through the marginal likelihood $P ( y _ { n } ; \mathbf { h } _ { n } )$ in (12), which itself involves an expectation over the posterior distribution from previous measurements. In practice, these integrations are often high-dimensional and lack closed-form expressions, relying on Monte Carlo-based approximations (see Section 3.4). This computation is repeated at each of the N sequential steps, making the full sOED approach (14) challenging for real-time or resource-constrained imaging applications.

These computational challenges motivate the reformulation of the sOED problem as a sequential decision-making under uncertainty, and the introduction of RL-based approaches as a principled way to tackle the sOED problem (14). Instead of solving the complex optimization problems behind sOED, one learns parametric policies that directly map posterior distributions to near-optimal measurement patterns by simulating a large number of sOED instances offline, i.e. before any real measurement takes place. The complex integrals only have to be computed during this offline training phase, not during deployment.

## 2.2.1 Sequential decision-making processes

Solving the sOED problem (14) can be seen as an instance of a Markov decision process (MDP), a mathematical model to describe sequential and time-discrete decision processes under uncertainty. In MDPs, an agent interacts dynamically with an environment in order to accumulate a reward: The state $S \in { \mathcal { S } }$ represents the current configuration of the agent and environment, and every possible decision the agent can make is represented by an action $A \in { \mathcal { A } }$ that will lead to a new state $S ^ { \prime }$ with a certain probability. The agent's decision making is represented by a policy $\pi ( S , A )$ , which is a probability distribution in A, encoding the likelihood that the agent takes the decision A when in state S. For every decision, the agent receives a reward $R ( S , A , S ^ { \prime } )$ . The goal in MDP is to find policies that maximize the accumulated rewards for a given time window. Very often, the agent does not have full knowledge of the state S, leading to a partially observable Markov decision process $( P O M D P )$ [16].

Several methodologies describe how to find a policy that maximizes the sum of rewards from the starting time $n = 1$ until the end of the experiment. In complex scenarios, obtaining analytical solutions for this objective or using conventional optimization schemes is often infeasible. Instead, RL methods achieve the optimization goal by updating the agent's policy based on the feedback provided by the environment at each time. The learning process accounts not only for exploiting stepwise successful actions, but also by exploring the landscape of possible environment states [16]. Two principal classes of RL algorithms exist: value-based methods and direct policy search methods. In value-based methods, the agent learns a scalar estimate of the expected cumulative rewards from a given state onward under a fixed policy, called the value function, from repeated environment interactions and derives a policy by acting greedily with respect to this function. In direct policy search, the policy is explicitly represented by a parametric function πθ and θ is updated directly via stochastic gradient ascent, i.e. policy gradient methods [17, 18].

In the sOED setting, however, the practical benefits of RL are only realized when the agent can be trained on a sufficiently large number of simulated trajectories, so that it can repeatedly experience the stochastic transitions $S  S ^ { \prime }$ and the associated rewards $R ( S , A , S ^ { \prime } )$ under many different policies. If the available simulation budget is too small, the agent cannot adequately explore the joint state-action space, and the resulting policy is prone to overfitting to a few sampled scenarios rather than approximating the truly optimal policy of the underlying MDP/POMDP, which substantially limits the applicability of RL-based sOED in practice.

## 3 Methods

## 3.1 SPI-sOED reinforcement learning framework

In this section, the sequential optimal experimental design problem for single-pixel imaging defined in Eq. (14) is formulated as a RL task and a methodology to solve it is developed. The key insight is that one must choose a complete sequence of patterns that jointly maximizes the cumulative information gain about the unknown image over the entire experiment, accounting for how each measurement shapes future decisions.

The Bayesian framework from Sec. 2.1 induces this sequential structure: At any measurement step n, the current knowledge about the image x is summarized by the posterior distribution $P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } )$ , where $\mathbf y _ { n - 1 }$ contains all measurements up to step n — 1 and ${ \bf H } _ { n - 1 }$ contains all previously used patterns. This reduces to the prior $P ( \mathbf { x } )$ when $n = 0$ (see Equation (5)). When a new pattern $\mathbf { h } _ { n }$ is chosen and measurement $y _ { n }$ is obtained, Bayes' rule gives us an updated posterior $P ( \mathbf { x } \mid \mathbf { y } _ { n } ; \mathbf { H } _ { n } )$ , which represents the refined knowledge. This sequential updating process can be outlined using the following decision-theoretic components:

• State: The posterior distribution $P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } )$ after measurement $n - 1$ serves as the current state $S _ { n }$ . This state encodes all information from past measurements and patterns that is relevant for future decisions.

• Action: The binary measurement pattern $\mathbf { h } _ { n }$ used at step n becomes the action $A _ { n }$ . It leads to the scalar measurement $y _ { n } = \mathbf { h } _ { n } ^ { \top } \mathbf { x } + \epsilon _ { n }$

• State transition: The new posterior distribution $P ( \mathbf { x } \mid \mathbf { y } _ { n } ; \mathbf { H } _ { n } )$ after measurement n becomes the next state $S _ { n + 1 }$ . This transition is stochastic because the measurement $y _ { n }$ is corrupted by noise $\epsilon _ { n } .$ but the Bayesian update rule provides the deterministic mapping from $( S _ { n } , A _ { n } , y _ { n } )$ to $S _ { n + 1 }$

• Reward: The immediate reward for choosing pattern $\mathbf { h } _ { n }$ in state $S _ { n }$ is the expected information gain $R _ { n } : = \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } )$ , defined in Eq. (9). Note that this reward depends only on the current state $S _ { n }$ and action $A _ { n }$ , and not on the resulting new state $S _ { n + 1 }$ . Computing EIG requires marginalizing over all possible outcomes $y _ { n }$ , which is only tractable in small-scale settings as the one showcased on this work. For more complex settings, $R _ { n }$ can be approximated by the actual information gain $\mathrm { I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } , y _ { n } )$ , which depends on the observed $y _ { n }$ and thereby $S _ { n + 1 }$ , cf. Section 3.4.

• Episode: A complete sequence of N measurement steps forms an episode or trajectory $\tau : = \left( S _ { 0 } , A _ { 1 } , S _ { 1 } , R _ { 1 } , \ldots , S _ { N - 1 } , A _ { N } , S _ { N } , R _ { N } \right)$ , representing one complete imaging experiment from prior to final posterior.

• Policy: The stochastic policy $\pi ( A _ { n } \mid S _ { n } )$ maps the current state, the posterior distribution over x, to a probability distribution over binary measurement patterns.

With these definitions, the sequential optimal design problem of maximizing the cumulative expected information gain from Eq. (14) can be recast as finding an optimal policy $\pi ^ { * }$ that maximizes the total reward

$$
\begin{array} { r l } & { \pi ^ { * } : \ = \underset { \pi } { \arg \operatorname* { m a x } } \mathbb { E } _ { \tau \sim P ( \tau | \pi ) } \Bigg [ \underset { n = 1 } { \overset { N } { \sum } } R _ { n } \Bigg ] } \\ & { \qquad = \underset { \pi } { \arg \operatorname* { m a x } } \mathbb { E } _ { \tau \sim P ( \tau | \pi ) } \Bigg [ \underset { n = 1 } { \overset { N } { \sum } } \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } ) \Bigg ] } \\ & { \qquad = : \underset { \pi } { \arg \operatorname* { m a x } } J ( \pi ) } \end{array}\tag{17}
$$

where $J ( \pi )$ is defined as the objective function for the policy training. The expectation in (17) over the distribution of trajectories $\tau \sim P ( \tau \mid \pi )$ integrates the stochasticity of the sample, policy and the observation noise. A trajectory can be generated as follows: a sample is drawn from the prior $\mathbf { x } \sim P ( \mathbf { x } )$ , set $S _ { 0 } = P ( x )$ and iterate for $n = 1 , \ldots , N$

$$
\begin{array} { r l } & { A _ { n } = \mathbf { h } _ { n } \sim \pi ( \cdot \mid S _ { n - 1 } ) , } \\ & { y _ { n } \sim P ( y _ { n } \mid \mathbf { x } ; \mathbf { h } _ { n } ) , } \\ & { S _ { n } = P ( \mathbf { x } \mid \mathbf { y } _ { n } ; \mathbf { H } _ { n - 1 } ) \propto P ( y _ { n } \mid \mathbf { x } ; \mathbf { h } _ { n } ) P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } ) , } \\ & { R _ { n } = \mathrm { E I G } _ { \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } ) = \mathbb { E } _ { P ( y _ { n } ; \mathbf { h } _ { n } ) } [ \mathrm { I G } _ { \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( y _ { n } ; \mathbf { h } _ { n } ) ] . } \end{array}\tag{18}
$$

The remaining question is how to represent and optimize the policy π in practice. The sOED state space consists of full posterior distributions over x, which are high-dimensional and continuous, making value function estimation intractable. This motivates the use of direct policy search methods, where $\pi$ is parameterized by a differentiable function $\pi _ { \theta }$ and optimized end-to-end via policy gradient ascent on $J ( \theta )$ . In the following sections, the specific parameterizations of $\pi$ adopted for SPI-sOED and the resulting learning algorithm are detailed.

## 3.2 SPI design policy parameterizations

The fundamental question is how to represent this distribution over binary patterns. To construct a policy from the current posterior state, we use a feed-forward neural network that maps a state representation $S _ { n - 1 }$ to a probability distribution over binary measurement patterns. We consider two policy parameterizations that share the same multilayer perceptron (MLP) backbone but differ in their output dimension and final activation function. The MLP follow the model

$$
f _ { \theta } : = w _ { \theta _ { L - 1 } } \circ \sigma _ { L - 1 } \circ . . . \circ \sigma _ { 1 } \circ w _ { \theta _ { 1 } } ,\tag{19}
$$

where w denotes an affine linear mapping with parameters $\theta _ { l } , ~ \sigma _ { l } ( \cdot )$ a non-linear activation function and o the composition of functions such that $( f \circ g ) ( \cdot ) = f ( g ( \cdot ) )$ , with $l = 1 , \ldots , L - 1$ the layer index.

A categorical policy $\pi _ { \boldsymbol { \theta } } ^ { C }$ directly outputs a probability distribution over all $2 ^ { M }$ possible patterns. With an output layer of $2 ^ { M }$ units and a Softmax activation, the probability for a possible pattern $\mathbf { h } _ { i }$ given a particular past state $S _ { n }$ -1 is

$$
\pi _ { \theta } ^ { C } ( \mathbf { h } _ { i } \vert S _ { n - 1 } ) = \frac { e ^ { f _ { \theta } ( S _ { n - 1 } ) _ { i } } } { \sum _ { j } e ^ { f _ { \theta } ( S _ { n - 1 } ) _ { j } } } ,\tag{20}
$$

where $j = 1 , \dots , 2 ^ { M }$ indexes all possible patterns. This approach allows the network to learn arbitrary correlations between pixel activations. However, it scales exponentially in M, making it practical only for very small images.

Alternatively, each pixel activation can be modeled independently: with M output units and a sigmoid activation $\sigma _ { L }$ , we obtain independent Bernoulli probabilities for the activation of each pixel $h _ { k } , k = 1 , \ldots , M$ of a pattern h

$$
\pi _ { \theta } ^ { B } ( h _ { k } = 1 | S _ { n - 1 } ) = \mathrm { B e r n o u l l i } \left( \frac { 1 } { 1 + e ^ { - f _ { \theta } ( S _ { n - 1 } ) _ { k } } } \right) .\tag{21}
$$

To sample a pattern, independent draws are sampled for each pixel from its corresponding Bernoulli distribution. This scales linearly in M, making it practical for larger images. Although the network still can learn to modulate individual probabilities based on the full posterior state, it cannot model correlations between pixel activations directly. With these state representations in place, the architecture of the input layer is determined by how the state is encoded, since the model processes that representation in a compatible form as input.

## 3.3 State representation and approximation

The RL framework from Sec. 3.1 requires at each step n a state $S _ { n }$ merging the current knowledge about the image and a reward $R _ { n }$ quantifying information gained. In the $\mathrm { S P I - s O E D }$ setting, both quantities are obtained via the Bayesian posterior

$$
P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } ) ,\tag{22}
$$

but their computation depends on whether this posterior is available in closed-form or only implicitly through sampling. As we introduced in Section 2.2.1, using a stochastic variable as a state corresponds to a partially observable Markov decision process.

Explicit posterior (closed-form state). When the posterior admits a closed-form expression, the full distribution reads

$$
S _ { n } ~ = ~ P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } )\tag{23}
$$

as the state.

A typical example is a linear Gaussian model with prior $\mathbf { x } \sim \mathcal { N } ( \pmb { \mu } _ { 0 } , \pmb { \Sigma } _ { 0 } )$ and likelihood

$$
y _ { n } \mid \mathbf { x } , \mathbf { h } _ { n } \sim { \mathcal { N } } ( \mathbf { h } _ { n } ^ { \top } \mathbf { x } , \sigma ^ { 2 } ) ,\tag{24}
$$

for which the posterior is again Gaussian,

$$
\pmb { \Sigma } _ { n } = \Big ( \pmb { \Sigma } _ { n - 1 } ^ { - 1 } + \sigma ^ { - 2 } \mathbf { h } _ { n } \mathbf { h } _ { n } ^ { \top } \Big ) ^ { - 1 } ,\tag{25}
$$

$$
{ \pmb \mu } _ { n } = \Sigma _ { n } \Big ( \Sigma _ { n - 1 } ^ { - 1 } { \pmb \mu } _ { n - 1 } + \sigma ^ { - 2 } y _ { n } { \bf h } _ { n } \Big ) ,\tag{26}
$$

with $( \pmb { \mu } _ { n } , \pmb { \Sigma } _ { n } )$ fully characterising $S _ { n }$ . In this regime, both the state and the expected information gain can be evaluated exactly, providing a high-fidelity learning signal.

Implicit posterior (sample-based state). For more complex priors or likelihoods, the posterior $P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } , \mathbf { H } _ { n - 1 } )$ is not available in closed-form and must instead be represented implicitly via a collection of posterior samples $\{ \mathbf { x } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { s } } \sim P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } , \mathbf { H } _ { n - 1 } )$ . Among several possible ways to encode this implicit representation as a state, two natural choices are considered. The first is to pass a set of samples directly to the policy,

$$
\hat { S } _ { n } = \{ \mathbf { x } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { s } } ,\tag{27}
$$

which preserves the complete distributional information at the cost of a fixed-size input of dimension $N _ { s } \times M$ . The second, computationally cheaper alternative is to summarize the set by its empirical mean and covariance analogues of the sufficient statistics $( \pmb { \mu } _ { n } , \pmb { \Sigma } _ { n } )$ from the closed-form Gaussian case in Eqs. (26), (25)

$$
\hat { \pmb { \mu } } _ { n } : = \frac { 1 } { N _ { s } } \sum _ { i = 1 } ^ { N _ { s } } { \bf x } ^ { ( i ) } ,\tag{28}
$$

$$
\hat { \mathbf { \Sigma } } _ { n } : = \mathbf { \Sigma } \frac { 1 } { N _ { s } - 1 } \sum _ { i = 1 } ^ { N _ { s } } \bigl ( \mathbf { x } ^ { ( i ) } - \hat { \pmb { \mu } } _ { n } \bigr ) \bigl ( \mathbf { x } ^ { ( i ) } - \hat { \pmb { \mu } } _ { n } \bigr ) ^ { \top } .\tag{29}
$$

which compresses the posterior into a low-dimensional pair of Gaussian moments that can be processed efficiently by the policy network. We study the numerical impact of the sample size $N _ { s }$ on policy learning in Sec. 4 for the one-active-pixel prior.

## 3.4 Reward computation and approximation: Noiseless and noisy cases

The EIG reward $R _ { n } = \operatorname { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } )$ is evaluated at each step of every training trajectory. Computing it exactly requires integrating over both the unknown image x and the future observation $y _ { n }$ , which is tractable only under specific model assumptions. Two variants appear as the posterior $P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } )$ and the likelihood $P ( y _ { n } \mid \mathbf { x } ; \mathbf { h } _ { n } )$ admit closed-form representations or not.

For a general prior $P ( \mathbf { x } )$ and likelihood $P ( y _ { n } \mid \mathbf { x } ; \mathbf { h } _ { n } )$ , the exact reward is

$$
\begin{array} { r l } & { R _ { n } = \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } } ( \mathbf { h } _ { n } ) } \\ & { \quad \quad = \mathrm { E n t } \big [ P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } ) \big ] - \mathbb { E } _ { y _ { n } } \Big [ \mathrm { E n t } \big [ P ( \mathbf { x } \mid y _ { n } , \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n } ) \big ] \Big ] , } \end{array}\tag{30}
$$

where the expectation is over the marginal $\begin{array} { r } { P ( y _ { n } ; \mathbf { h } _ { n } ) = \int _ { \mathcal { X } } P ( y _ { n } \mid \mathbf { x } ; \mathbf { h } _ { n } ) P ( \mathbf { x } \mid } \end{array}$ ${ \bf y } _ { n - 1 } ; { \bf H } _ { n - 1 } )$ dx. In general, evaluating (30) exactly is intractable because it requires integrating over both x and $y _ { n }$

Given $N _ { \mathrm { r } }$ posterior samples $\{ \mathbf { x } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { r } } } \sim P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } )$ one can approximate the marginal as

$$
\hat { P } ( y _ { n } ; { \bf h } _ { n } ) = \frac { 1 } { N _ { \mathrm { r } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r } } } P \Big ( y _ { n } \mid { \bf x } ^ { ( i ) } ; { \bf h } _ { n } \Big ) .\tag{31}
$$

To estimate the EIG itself, a single held-out sample is drawn $\mathbf { x } ^ { ( 0 ) } \sim$ $P ( \mathbf { x } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } )$ , independent of the $N _ { \mathrm { r } }$ denominator particles, and use the nested Monte Carlo (NMC) estimator [19]

$$
\hat { R } _ { n } = \int _ { \mathcal { Y } } P ( y _ { n } \mid \mathbf { x } ^ { ( 0 ) } ; \mathbf { h } _ { n } ) \left[ \log P ( y _ { n } \mid \mathbf { x } ^ { ( 0 ) } ; \mathbf { h } _ { n } ) - \log \hat { P } ( y _ { n } ; \mathbf { h } _ { n } ) \right] d y _ { n } .\tag{32}
$$

The one-dimensional integral in (32) is over the observation space $\mathcal { V }$ and can be evaluated analytically or by quadrature depending on the form of the likelihood.

In in absence of noise, i.e. $y _ { n } = \left. \mathbf { h } _ { n } , \mathbf { x } \right.$ , the likelihood collapses to a point mass $P ( y _ { n } \mid \mathbf { x } ; \mathbf { h } _ { n } ) = \delta { \big ( } y _ { n } - \langle \mathbf { h } _ { n } , \mathbf { x } \rangle { \big ) }$ . In this case the integral over $\mathcal { V }$ in (32) vanishes and the estimator reduces to a purely discrete sum over particles. Concretely, the forward model is evaluated at the held-out sample to obtain $y _ { n } ^ { ( 0 ) } = \langle { \bf h } _ { n } , { \bf x } ^ { ( 0 ) } \rangle$ and at each denominator particle to form the empirical marginal likelihood. Because $y _ { n }$ takes values in a finite set $\{ \langle \mathbf { h } _ { n } , \mathbf { x } ^ { ( i ) } \rangle \} _ { i = 0 } ^ { N _ { \mathrm { r } } } ,$ the EIG estimate becomes

$$
\hat { R } _ { n } ^ { \mathrm { n o i s e l e s s } } = - \log \left( \frac { 1 } { N _ { \mathrm { r } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r } } } \mathbf { 1 } \Big [ \big < \mathbf { h } _ { n } , \mathbf { x } ^ { ( i ) } \big > = \big < \mathbf { h } _ { n } , \mathbf { x } ^ { ( 0 ) } \big > \Big ] \right) ,\tag{33}
$$

where $\mathbf { 1 } [ \cdot ]$ is the indicator function. This is simply the negative log of the fraction of particles that predict the same observation as the held-out sample.

When the observation is corrupted by additive uncorrelated Gaussian noise, $y _ { n } = \left. \mathbf { h } _ { n } , \mathbf { x } \right. + \varepsilon$ with $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ , the likelihood becomes $P ( y _ { n } \mid$ $\mathbf { x } ; \mathbf { h } _ { n } ) = { \mathcal { N } } { \big ( } \langle \mathbf { h } _ { n } , \mathbf { x } \rangle , \sigma ^ { 2 } { \big ) }$ . One can approximate it via numerical quadrature on a uniform grid $\{ y _ { k } \} _ { k = 1 } ^ { K }$ spanning $\begin{array} { r } { [ \operatorname* { m i n } _ { i } \langle \mathbf { h } _ { n } , \mathbf { x } ^ { ( i ) } \rangle - 5 \sigma , } \end{array}$ $\operatorname* { m a x } _ { i } \langle \mathbf { h } _ { n } , \mathbf { x } ^ { ( i ) } \rangle + 5 \sigma ]$ with spacing $\Delta y$ This quadrature scheme can be viewed as a deterministic variant of the nested Monte Carlo (NMC) estimator of [20], where the outer expectation over observations is resolved on a fixed grid rather than by sampling; this removes the variance of the outer loop at the cost of scaling linearly with the grid resolution $K$ , which remains tractable when the observation space is one-dimensional.

At each grid point $y _ { k }$ , define the weighted likelihood as

$$
q _ { i k } = P ( \mathbf { x } ^ { ( i ) } \mid \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } ) \cdot { \mathcal { N } } { \Big ( } y _ { k } ; \langle \mathbf { h } _ { n } , \mathbf { x } ^ { ( i ) } \rangle , \sigma ^ { 2 } { \Big ) } ,\tag{34}
$$

the marginal $\begin{array} { r } { m _ { k } = \sum _ { i = 1 } ^ { N _ { \mathrm { x } } } q _ { i k } } \end{array}$ , and the posterior weights:

$$
P ( { \bf x } ^ { ( i ) } \mid y _ { k } , { \bf y } _ { n - 1 } ; { \bf H } _ { n - 1 } , { \bf h } _ { n } ) = \frac { q _ { i k } } { m _ { k } } ,\tag{35}
$$

where · is the scalar product.

The expected posterior entropy is obtained by averaging the entropy of the posterior distribution over the observation space via quadrature:

$$
\hat { \mathbb { E } } _ { y _ { n } } [ \mathrm { E n t } \big [ P ( \mathbf { x } \mid y _ { n } , \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n } ) \big ] \big ] = \sum _ { k = 1 } ^ { K } m _ { k } \mathrm { E n t } \big [ P ( \mathbf { x } \mid y _ { k } , \mathbf { y } _ { n - 1 } ; \mathbf { H } _ { n - 1 } , \mathbf { h } _ { n } ) \big ] \Delta y ,\tag{36}
$$

where $m _ { k }$ acts as the quadrature weight, combining the marginal density with the grid spacing. The expected information gain is then

$$
\hat { R } _ { n } ^ { \mathrm { G a u s s i a n } } = \mathrm { E n t } [ P ( { \bf x } \mid { \bf y } _ { n - 1 } ; { \bf H } _ { n - 1 } ) ] - \hat { \mathbb { E } } _ { y _ { n } } [ \mathrm { E n t } \big [ P ( { \bf x } \mid y _ { n } , { \bf y } _ { n - 1 } ; { \bf H } _ { n } ) \big ] ] .\tag{37}
$$

## 3.4.1 Gaussian Prior case: D-Optimal Sequential Criterion

When the prior is Gaussian, $P ( \mathbf { x } ) = \mathcal { N } ( \mathbf { x } ; \pmb { \mu } _ { 0 } , \pmb { \Sigma } _ { 0 } )$ , the forward operator is linear and the noise is Gaussian, the posterior remains Gaussian at every step n with mean $\pmb { \mu } _ { n - 1 }$ and covariance $\Sigma _ { n - 1 }$ updated via the recursions in (25) and (26). In this setting, the model evidence for the n-th observation is itself Gaussian, $P ( y _ { n } \mid \mathbf { h } _ { n } ) = { \mathcal { N } } ( y _ { n } ; \mathbf { h } _ { n } ^ { \top } \pmb { \mu } _ { n - 1 } , s _ { n } ^ { 2 } )$ , with predictive variance $s _ { n } ^ { 2 } = \mathbf { h } _ { n } ^ { \top } \pmb { \Sigma } _ { n - 1 } \mathbf { h } _ { n } + \sigma ^ { 2 }$ . Since the posterior covariance $\Sigma _ { n }$ is independent of the observed value $y _ { n }$ , averaging over the model evidence via Gaussian moment identities yields a closed-form reward

$$
R _ { n } ^ { \mathrm { G a u s s } } ( \mathbf { h } _ { n } ) = \frac { 1 } { 2 } \log \operatorname* { d e t } \biggr ( \mathbf { I } + \sigma ^ { - 2 } \boldsymbol { \Sigma } _ { n - 1 } ^ { 1 / 2 } \mathbf { h } _ { n } \mathbf { h } _ { n } ^ { \top } \boldsymbol { \Sigma } _ { n - 1 } ^ { 1 / 2 } \bigr ) = \frac { 1 } { 2 } \log \biggr ( 1 + \frac { \mathbf { h } _ { n } ^ { \top } \boldsymbol { \Sigma } _ { n - 1 } \mathbf { h } _ { n } } { \sigma ^ { 2 } } \biggr ) ,\tag{38}
$$

where the second equality follows from the matrix determinant lemma [21].

This is precisely the D-optimal design criterion evaluated sequentially [22, 23]: the optimal pattern $\mathbf { h } _ { n } ^ { * }$ at step n maximises the predictive variance $ { \mathbf { h } } _ { n } ^ { \top } \pmb { \Sigma } _ { n - 1 }  { \mathbf { h } } _ { n }$ of the current Gaussian belief, automatically adapting to all past measurements through the running covariance $\Sigma _ { n - 1 }$ . The total EIG over N steps then decomposes additively as

$$
\mathrm { E I G } ( \mathbf { H } _ { N } ) = \sum _ { n = 1 } ^ { N } { R } _ { n } ^ { \mathrm { G a u s s } } ( \mathbf { h } _ { n } ) = \frac { 1 } { 2 } \log \operatorname* { d e t } \Bigl ( \mathbf { I } + \sigma ^ { - 2 } \boldsymbol { \Sigma } _ { 0 } ^ { 1 / 2 } \mathbf { H } _ { N } ^ { \top } \mathbf { H } _ { N } \boldsymbol { \Sigma } _ { 0 } ^ { 1 / 2 } \Bigr ) ,\tag{39}
$$

recovering the standard Bayesian D-optimality objective [22].

Unlike the general estimator in (37), (39) requires no outer quadrature over  and no posterior samples, only the covariance update, making it exact and computationally negligible.

## 3.5 Pattern design including budget constraints

The framework developed so far optimises binary measurement patterns to maximise information gain, without placing any other explicit restrictions on them. In many real-world applications, however, sensing is not free: every observation consumes a physical resource, as time, energy, or radiation dose, whose accumulation can harm the very system being studied. In live-cell fluorescence microscopy, for instance, cumulative light exposure directly damages the cells through multiple photodamage mechanisms [24, 25]: excitation light generates reactive oxygen species that degrade cellular components, induces phototoxic stress responses altering gene expression, and accelerates photobleaching of fluorophores. The severity of these effects scales with total photon dose, which is simply the $\ell _ { 1 , 1 }$ norm of ${ \bf { H } } _ { N }$ . The dose-constrained sequential optimal experimental design for single-pixel imaging is defined as

$$
\pi _ { B } ^ { * } : = \arg \operatorname* { m a x } _ { \pi } J ( \pi ) \mathrm { \quad ~ s . t . ~ } \sum _ { i , j } ( { \bf H } _ { N } ) _ { i , j } \leq B
$$

The budget B encourages the policy $\pi _ { B } ^ { * }$ to focus sensing capacity on the most informative regions, naturally promoting sparse, adaptive measurement strategies.

One approach to enforce the hard constraint is to directly optimize over policies that fulfill it, e.g., by using the conditional Bernoulli policies $\pi _ { \theta , B } ^ { C B e r } ( A _ { n } \ | \ S _ { n - 1 } )$ [26], which generates each new pattern conditioned on the dose used so far. While using these policies guarantees strict dose compliance, it introduces sequential dependencies across pattern proposals, which can slow training and requires the policy to anticipate future sensing opportunities.

An alternative is to fold the cost into the objective as a soft constraint, penalizing dose expenditure directly within the policy gradient

$$
\pi _ { \lambda } ^ { * } : = \arg \operatorname* { m a x } _ { \pi } J ( \pi ) - \lambda \sum _ { i , j } \left( \mathbf { H } _ { N } \right) _ { i , j }\tag{40}
$$

where $\lambda > 0$ balances information gain against dosage used. This $\mathrm { L a } -$ grangian relaxation allows more flexible pattern proposals and avoids the sequential conditioning required by the hard-constrained approach, but does not guarantee strict dose compliance: A too small λ may lead to B routinely exceeded; too large λ and informative patterns may be suppressed.

## 3.6 Learning experimental designs using parametric policies

For policy gradient methods, improving the experimental design reduces to iteratively updating the policy parameters θ introduced in Sec. 3.2 to maximize the expected information gain objective $J ( \pi _ { \theta } ) = J ( \theta )$ by gradient ascent

$$
\theta _ { n } = \theta _ { n - 1 } + \alpha \nabla _ { \theta } J ( \theta _ { n } )\tag{41}
$$

where $\alpha$ is learning rate of the parameterization update.

As computing the true gradient is intractable, different methods rely on different stochastic approximation approaches. When a differentiable policy is available, one can make use of the gradient policy theorem [27], to reformulate the gradient of (17) as:

$$
\begin{array} { r l } { \nabla _ { \theta } J ( \theta ) = \nabla _ { \theta } \mathbb { E } _ { \tau \sim P ( \tau \mid \pi _ { \theta } ) } \Bigg [ \displaystyle \sum _ { n = 1 } ^ { N } R _ { n } ( S _ { n - 1 } , A _ { n } , S _ { n } ) \Bigg ] } & { } \\ { = \mathbb { E } _ { \tau \sim P ( \tau \mid \pi _ { \theta } ) } \Bigg [ \displaystyle \sum _ { n = 1 } ^ { N } G _ { n } \nabla _ { \theta } \log \pi _ { \theta } ( A _ { n } \mid S _ { n - 1 } ) \Bigg ] \quad , } \end{array}\tag{42}
$$

where $\begin{array} { r } { G _ { n } : = \sum _ { l = n } ^ { N } R _ { l } } \end{array}$ is the expected cumulative reward.

At each step n, the update is proportional to $\begin{array} { r } { G _ { n } = \sum _ { l = n } ^ { N } R _ { l } } \end{array}$ , the sum of all rewards collected from that step until the end of the experiment. This means the policy is geared towards choices that proved beneficial across the remainder of the episode. Actions that consistently produce rewards are reinforced, while those that led to diminishing returns are gradually discarded. $\mathrm { B y }$ doing this, the expected cumulative return attributes longterm success back to the decisions that were crucial in the past.

The policy gradient update is the core step in the REINFORCE algorithm [16], which approximates the expected value in (42) by Monte Carlo estimation over $N _ { M C }$ trajectory samples: For each trajectory sample, the expected cumulative reward is computed for each step from that point until $n = N$ and using (42) the gradient in (41) is approximated as

$$
\nabla _ { \theta } J ( \theta ) ~ \approx \frac { 1 } { N _ { M C } } \sum _ { j = 1 } ^ { N _ { M C } } \sum _ { n = 1 } ^ { N } G _ { n } ^ { ( j ) } \nabla _ { \theta } \log \pi _ { \theta } ( A _ { n } ^ { ( j ) } \mid S _ { n - 1 } ^ { ( j ) } )\tag{43}
$$

The full REINFORCE algorithm formulated for the sOED problem is described in Algorithm 1.

Algorithm 1 REINFORCE for SPI Sequential Optimal Experimental De  
sign   
1: Initialize policy parameters θ randomly   
2: for episode $i = 1$ to $N _ { \mathrm { e p i s o d e s } }$ do   
3: sample ground truth image $\mathbf { x } \sim P ( \mathbf { x } )$ for this episode   
4: $/ /$ Collect $N _ { M C }$ trajectories   
5: for trajectory $j = 1$ to $N _ { M C }$ do   
6: $\tau ^ { ( j ) } \gets \emptyset$   
7: $S _ { 0 } ^ { ( j ) }  P ( \mathbf { x } )$ (prior distribution evaluated at x)   
8: for step $n = 1$ to N do   
9: $\mathbf { h } _ { n } ^ { ( j ) } \sim \pi _ { \theta } ( \cdot \mid S _ { n - 1 } ^ { ( j ) } )$   
10: $y _ { n } ^ { ( j ) } \sim P ( y \mid \mathbf { x } ; \mathbf { h } _ { n } ^ { ( j ) } )$   
11: $R _ { n } ^ { ( j ) } = \mathrm { E I G } _ { \mathbf { x } | \mathbf { y } _ { n - 1 } ^ { ( j ) } ; \mathbf { H } _ { n - 1 } ^ { ( j ) } } ( \mathbf { h } _ { n } ^ { ( j ) } )$ as described in Sec. 3.4   
12: $S _ { n } ^ { ( j ) } \gets P ( \mathbf { x } \mid \mathbf { y } _ { n } ^ { ( j ) } ; \tilde { \mathbf { H } } _ { n } ^ { ( j ) } ) \propto P ( y _ { n } ^ { ( j ) } \mid \mathbf { x } ; \mathbf { h } _ { n } ^ { ( j ) } ) \cdot S _ { n - 1 } ^ { ( j ) }$ as described   
in Sec. 3.3   
13: $\tau ^ { ( j ) }  \tau ^ { ( j ) } \cup \{ ( S _ { n - 1 } ^ { ( j ) } , \mathbf { h } _ { n } ^ { ( j ) } , R _ { n } ^ { ( j ) } , S _ { n } ^ { ( j ) } ) \}$ as described in Sec. 3.1   
14: end for   
15:   
16: $/ /$ Compute expected cumulative reward   
17: for step $n = 1$ to N do   
18: $\begin{array} { r } { G _ { n } ^ { ( j ) }  \sum _ { l = n } ^ { N } R _ { l } ^ { ( j ) } } \end{array}$ (sum of rewards from step n)   
19: end for   
20: end for   
21:   
22: //Update policy using (43)   
23: $\begin{array} { r } { \overset { \cdot \cdot } { \nabla } _ { \theta } \overset { \cdot } { J } ( \theta )  \frac { 1 } { N _ { M C } } \sum _ { j = 1 } ^ { N _ { M C } } \sum _ { n = 1 } ^ { \bar { N } } G _ { n } ^ { ( j ) } \cdot \nabla _ { \theta } \log \pi _ { \theta } ( A _ { n } ^ { ( j ) } = \mathbf { h } _ { n } ^ { ( j ) } \mid S _ { n - 1 } ^ { ( j ) } ) } \end{array}$   
24: Update parameters $\theta  \theta + \alpha \cdot \nabla _ { \theta } J ( \theta )$ as in (41)   
25: end for   
26: return Trained policy πθ

## 4 Numerical Studies

The proposed reinforcement learning approach for sOED is validated by learning optimal measurement strategies in a simple, yet representative, numerical SPI scenario. Concepts introduced in the previous section and their impact on learning performance, such as the policy parametrization (cf. Sec, 3.6), state approximation (cf. Sec 3.3), reward approximation (cf. Sec 3.4), dose constraints (cf. Sec. 3.5) performance are illustrated. The ability of the framework to find optimal strategies under increasing measurement noise is shown.

![](images/c3d94bdbb64a59900c859a076cd677a193ad2ff2de66f09f6e7a9cd0f08f59a1.jpg)  
Figure 1: The posterior is being updated using measurements of the ground truth x via the proposed patterns $\mathbf { h } _ { n }$ (in green). Bisection and raster scan are considered as two possible strategies to identify the ground truth from measurements via one-active-pixel prior knowledge.

## 4.1 One-active-pixel SPI scenario

Consider a structured single-pixel imaging setup in which exactly one pixel of the unknown image is active with intensity 1, as described in the left side of Figure 1. This discrete sparsity-constraint means that the possible images are reduced to the discrete set of canonical basis vectors, and one can assume that the prior $P ( \mathbf { x } )$ accurately captures this a-priori information. In this setting, both the Bayesian posterior and the expected information gain are tractable to compute either in closed form, for the noiseless scenario, or via low-dimensional numerical quadrature in the noisy case. It provides a simple benchmark in which the reinforcement learning agent can use the exact posterior as the state and the sensing strategy can be compared against analytically optimal designs as well as against random sensing pattern. It also enables a systematic study of variability across independent policy initializations, since the discrete prior supports repeated runs with a modest computational cost.

Formally, the one-active-pixel set can be described by the images $\mathcal { E } _ { M } : =$ $\{ \mathbf { e } _ { 1 } , \hdots , \mathbf { e } _ { M } \}$ , where $\mathbf { e } _ { m }$ denotes the m-th unit vector in $\mathbb { R } ^ { M }$ . The prior $P ( \mathbf { x } )$ is a discrete probability measure on $\mathbb { R } ^ { M }$ and can be written as

$$
P ( { \bf x } ) = \sum _ { m = 1 } ^ { M } w _ { m } ^ { ( 0 ) } \delta _ { { \bf e } _ { m } } ( { \bf x } ) , \qquad w _ { m } ^ { ( 0 ) } = 1 / M\tag{44}
$$

where $\delta _ { \mathbf { e } _ { m } } ( \mathbf { x } )$ denotes a unit point mass concentrated at ${ \mathbf x } = { \mathbf e } _ { m }$ . Following (3), each measurement leads to posterior that is also discrete and has the same form as the prior, just with updated weights

$$
P ( \mathbf { x } \mid \mathbf { y } _ { n } ; \mathbf { H } _ { n } ) \ = \ \sum _ { m = 1 } ^ { M } w _ { m } ^ { ( n ) } \delta _ { \mathbf { e } _ { m } } ( \mathbf { x } ) , \quad w _ { m } ^ { ( n ) } \propto P ( \mathbf { y } _ { n } \mid \mathbf { x } = \mathbf { e } _ { m } ; \mathbf { H } _ { n } ) , \quad \sum _ { m = 1 } ^ { M } w _ { m } ^ { ( n ) } = 1\tag{45}
$$

Thus, the posterior and the RL state can be fully represented by the M-dimensional vector $S _ { n } = \mathbf { w } ^ { ( n ) } = ( w _ { 1 } ^ { ( n ) } , \dots , w _ { M } ^ { ( n ) } )$

In the ideal noiseless one-active-pixel setting, a measurement of any $\mathbf { x } =$ $\mathbf { e } _ { m }$ that is supported by the prior with a binary measurement pattern h $\in$ $\{ 0 , 1 \} ^ { M }$ yields either 0 or 1

$$
y = \langle \mathbf { h } , \mathbf { e } _ { m } \rangle = \left\{ { \begin{array} { l l } { 1 } & { { \mathrm { i f } } \qquad ( \mathbf { h } ) _ { m } = 1 } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{46}
$$

If one measures $y = 1$ , the active pixel then lies in the support of the illuminated mask, i.e., m ∈ supp(h). Updating the prior with this likelihood information means removing all one-active-pixel images with active locations outside of the mask and renormalizing the weights to yield a uniform distribution over the remaining locations, as illustrated in the trajectories of Figure 1. Measuring $y = 0$ accordingly removes all locations that are in the support of the illuminated mask from the posterior. Intuitively, this scenario thereby corresponds to a simple "finding a needle in a haystack" search problem: where, out of m possible locations, a particular objects sits. Every measurement one can make with a pattern h corresponds to asking the question "Is the object located in any of the illuminated positions?" and receiving $\mathrm { a \ ^ { 3 9 } y e s  – o r – n o ^ { 3 9 } }$ answer.

For illustration, let us consider the case $M = 4$ case: The prior is uniform over $\{ \mathbf { e } _ { 1 } , \ldots , \mathbf { e } _ { 4 } \}$ , leading to $\mathbf { w } ^ { ( 0 ) } = ( 1 / 4 , 1 / 4 , 1 / 4 , 1 / 4 )$ . The simplest case is the conventional raster scan, where one pixel at a time is illuminated in a fixed order. The sequence of patterns would be ${ \bf h } _ { 1 } = ( 1 , 0 , 0 , 0 )$ , h2 = $( 0 , 1 , 0 , 0 )$ , h3 = (0, 0, 1, 0), h4 $\mathbf { \Lambda } = \mathbf { \Lambda } ( 0 , 0 , 0 , 1 )$ . Now suppose that the true image is $\mathbf { x } = \mathbf { e } _ { 3 }$ . The first measurement is then $y _ { 1 } = 0$ , which discards $\mathbf { e } _ { 1 }$ but leaves $\mathbf { e } _ { 2 } , \mathbf { e } _ { 3 } , \mathbf { e } _ { 4 }$ as possible. After the likelihood update, the posterior weights become

$$
\mathbf { w } ^ { ( 1 ) } = \left( 0 , \ \frac { 1 } { 3 } , \ \frac { 1 } { 3 } , \ \frac { 1 } { 3 } \right) .\tag{47}
$$

The second pattern yields again $y _ { 2 } = 0$ , ruling out $\mathbf { e } _ { 2 }$ and leaving only $\mathbf { e } _ { 3 }$ and $\mathbf { e } _ { 4 }$ feasible, so the posterior weights are

$$
\mathbf { w } ^ { ( 2 ) } = \left( 0 , 0 , \frac { 1 } { 2 } , \frac { 1 } { 2 } \right) .\tag{48}
$$

Only after the third measurement, with ${ \bf h } _ { 3 } = ( 0 , 0 , 1 , 0 )$ and outcome $y _ { 3 } = 1$ , does the posterior collapses to a single delta distribution centered on the true image, such that weights are

$$
\mathbf { w } ^ { ( 3 ) } = ( 0 , 0 , 1 , 0 ) .\tag{49}
$$

In this worst-case trajectory, the posterior support shrinks from four to three, then to two, and only then to a single element, illustrating that, under a uniform prior, raster scanning requires on average $( M { + } 1 ) / 2$ measurements to uniquely identify the active pixel, exhibiting a linear measurement complexity in the number of image elements M.

In analogy to search problems, a bisection strategy would be more advantageous, where the patterns are chosen to half the current possible locations: In the same $M = 4$ example with uniform prior $\mathbf { w } ^ { ( 0 ) }$ and true image ${ \bf x } = { \bf e } _ { 3 }$ 2 one can consider a first pattern ${ \bf h } _ { 1 } = ( 1 , 1 , 0 , 0 )$ , which only illuminates pixels 1 and 2. The first outcome is then $y _ { 1 } = 0$ and after renormalizing the posterior becomes uniform over $\mathbf { e } _ { 3 }$ and $\mathbf { e } _ { 4 }$ so that the weights are

$$
\mathbf { w } ^ { ( 1 ) } = \Big ( 0 , 0 , \frac { 1 } { 2 } , \frac { 1 } { 2 } \Big ) ,\tag{50}
$$

An adaptive second pattern is then ${ \bf h } _ { 2 } = ( 0 , 0 , 1 , 0 )$ , which illuminates the third but not the fourth element. The second outcome is $y _ { 2 } = 1$ , collapsing to a delta centered in $\mathbf { e } _ { 3 }$ and the weights become

$$
\mathbf { w } ^ { ( 2 ) } = ( 0 , 0 , 1 , 0 ) ,\tag{51}
$$

shrinking from four to two and then directly to a single element in just two measurements.

Another possibility is the usage of a pattern that, on top of halving the possible locations, also keeps illuminating a partial fraction of the already measured pixels. From the previous case, a pattern ${ \bf h } _ { 2 } = ( 0 , 1 , 1 , 0 )$ represents a fxed bisection strategy, and these strategies are different from the adaptive ones in the dose budget they require and their dependence on the measurements. Fixed bisection allows via redundancy to solve for all possible images in the one-active-pixel set without redesigning their activations given the measurements obtained over the experiment.

For both cases, bisection strategies half the support of the posterior distribution at each step and, more generally, identifies the active pixel in at most $\lceil \log _ { 2 } M \rceil$ measurements, reflecting its logarithmic complexity in this simplified SPI setting.

## 4.2 Optimal EIG strategies

For the scenario described, one can find that bisection strategies are optimal with respect to EIG. Using this result, learned policies can be compared against the strategies yielding the best achievable performance. Firstly the EIG (9) is computed for images in the one-active-pixel set $x \in \mathcal { E } \mathrm { ~ = ~ }$ $\{ \mathbf { e } _ { 1 } , \hdots , \mathbf { e } _ { M } \}$ for the first measurement with h. The marginal likelihood

![](images/76883b406fd14074e57aa55736482064eb706ad01e39c03b3b5705863a694d44.jpg)  
Figure 2: For a uniform one-active-pixel, strategies that illuminate either one or all but one site are suboptimal in an EIG sense. This can be seen from the unbalanced posterior at the different possible outcomes y. For the bisection strategy, measurements 0 or 1 have equal probability and the posterior support halves, which produces the optimal one-step EIG.

$P ( \mathbf { y } ; \mathbf { h } )$ is needed to compute the expectation over the prior and posterior entropy difference.

As illustrated in Figure 2, y can only take the values 0 and 1 in the noiseless scenario: The marginal likelihood of $y = 1$ is the ratio between the number of illuminated sites in the measuring pattern $h , \mid \operatorname { s u p p } ( \mathbf { h } )$ |and the number of possible illuminated sites M, so that $P ( y = 1 ; { \bf h } ) = \mid \mathrm { s u p p } ( { \bf h } ) \mid / M$ and $P ( y = 0 ; { \bf h } ) = ( M - \mid \mathrm { s u p p } ( { \bf h } ) \mid ) / M$

Then, the entropy for the uniform one-active-pixel prior, as described in (44), reads

$$
\operatorname { E n t } ( \mathbf { x } ) = \log M\tag{52}
$$

Following the previous section, the posterior shrinks with respect to the uniform prior. One can compute the posterior entropy by averaging over the possible images of the set after the measurement, which leaves

$$
\operatorname { E n t } ( \mathbf { x } \mid y ; \mathbf { h } ) = { \left\{ \begin{array} { l l } { \log ( M - \mid \operatorname { s u p p } ( \mathbf { h } ) \mid ) } & { i f \qquad y = 0 } \\ { \log ( \mid \operatorname { s u p p } ( \mathbf { h } ) \mid ) } & { i f \qquad y = 1 } \end{array} \right. }\tag{53}
$$

With this, one can write the one-step EIG as an information gain average weighted by the marginal likelihood $P ( y ; \mathbf { h } )$ . Decomposing the information gain as the difference between prior and posterior entropy, one obtains

$$
\begin{array} { l } { { \displaystyle \mathrm { E I G } _ { \mathbf { x } } ( { \bf h } ) = \sum _ { y = 0 , 1 } P ( y ; { \bf h } ) [ \mathrm { E n t } ( { \bf x } ) - \mathrm { E n t } ( { \bf x } \mid y ; { \bf h } ) ] } \ ~ } \\ { { \displaystyle ~ = \frac { \left| \mathrm { s u p p } ( { \bf h } ) \right| } { M } \log \frac { M } { \left| \mathrm { s u p p } ( { \bf h } ) \right| } + \frac { M - \left| \mathrm { s u p p } ( { \bf h } ) \right| } { M } \log \frac { M } { M - \left| \mathrm { s u p p } ( { \bf h } ) \right| } } , } \end{array}\tag{54}
$$

which means that $\mathrm { E I G } _ { \mathbf { x } } ( \mathbf { h } )$ is a function of $s = \left| \ \mathrm { s u p p } ( \mathbf { h } ) \ \right|$ . If s was a continuous variable on $( 0 , M )$

$$
\frac { d } { d s } \operatorname { E I G } _ { \mathbf { x } } ( s ) = \frac { 1 } { M } \log \frac { M - s } { s } , \qquad \frac { d ^ { 2 } } { d s ^ { 2 } } \operatorname { E I G } _ { \mathbf { x } } ( s ) = - \frac { 1 } { M } \Big ( \frac { 1 } { s } + \frac { 1 } { M - s } \Big ) < 0 ,\tag{55}
$$

so the $\operatorname { E I G } _ { \mathbf { x } } ( s )$ would be strictly concave w.r.t. s and its unique stationary point

$$
s ^ { * } = \arg \operatorname* { m a x } _ { s } \mathrm { E I G } _ { \mathbf { x } } ( s ) = \frac { M } { 2 }\tag{56}
$$

is the global maximum. This means that for an even M, the optimal pattern illuminates half the sites of the possible image set $\mathcal { E }$ in one step.

Because every posterior is again uniform, now over the remaining image set ${ \mathcal { E } } _ { i } \subseteq { \mathcal { E } }$ of size $M _ { i }$ with $M _ { 0 } = M$ , the step-(i+1) design problem is identical to the first with M corresponding to $M _ { i }$ and $\mathrm { s u p p } ( \mathbf { h } _ { i + 1 } ) \subseteq \mathcal { E } _ { i }$ . The same calculation gives $s _ { i + 1 } ^ { * } = M _ { i } / 2$ and at the maximum both outcomes leave $M _ { i } / 2$ possible sites. Thereby, $M _ { i + 1 } = M _ { i } / 2$ and the support halves at every step, leading to $M _ { i } = M / 2 ^ { i }$ . Bisection is then EIG-optimal per step and identifies the active pixel in exactly $\lceil \log _ { 2 } M \rceil$ measurements no matter where the active pixel is located.

## 4.3 Experimental settings

In all presented RL experiments, $N = 2$ is set as the number of measurements, so that the posterior is updated twice per experiment. Furthermore, $N _ { M C } = 1$ is always used in (43), i.e., the policy update is based on a single simulated trajectory. This corresponds to the setting one would use for large-scale imaging scenarios. Every RL experiment is repeated 100 times with different policy initializations to study the stability and convergence of learned designs and report the mean and 95% confidence intervals in the evaluations. Unless stated otherwise, the total expected information gain of a policy $\pi \theta$ is normalized for visualization purposes, to allow for the comparison of policies that propose binary pattern strategies under different experimental setups

$$
\mathrm { J } ( \pi _ { \theta } ) = \frac { J ( \pi _ { \theta } ) - \overline { { { \mathrm { J } } } } _ { \mathrm { r a n d } } } { \mathrm { J } _ { \mathrm { o p t } } - \overline { { { \mathrm { J } } } } _ { \mathrm { r a n d } } } \qquad ,\tag{57}
$$

![](images/dff15837b8ad1e3fe6693f676e7141ae9b69147b8069e32080a1c7c8ad62ff7e.jpg)  
Figure 3: Total normalized EIG for categorical (lighter tone, square marker) and Bernoulli (darker tone, circle marker) policies trained over different image sizes $M = \{ 4 , 8 , 1 2 , 1 6 \}$ and same experimental step size $N \ = \ 2$ Optimal and random baselines are drawn as red and $\mathrm { g r a y }$ lines, respectively.

where the maximum EIG value $\mathrm { J _ { o p t } }$ is calculated for that experiment, obtained using a bisection strategy in the one-active-pixel prior case, and the average $\mathrm { E I G ~ \overline { { J } } _ { r a n d } }$ that a fully random policy reaches which illuminates each pixel with probability 0.5.

## 4.4 Policy architecture comparison

Figure 3 shows two training comparisons: the scaling of image sizes $M =$ {4, 8, 12, 16} and the usage of categorical or Bernoulli pattern proposal architectures, as described in Section 3.2. Categorical-based policies, described in (20), reach optimal convergence in fewer episodes than the Bernoulli-based ones, formulated in (21). Scaling the image size shows a higher requirement of episodes for convergence for the optimal EIG in two experimental steps.

The normalized EIG at Figure 3 shows that Bernoulli policies are slower to train than categorical policies for all image sizes. However, Bernoulli policies can parameterize distributions over the action space with fewer trainable parameters, as the output of the network is the distribution over the pattern activations, and this output layer grows linearly with the image size M. Actions sampled from the Bernoulli policies are rewarded indirectly, since there is not an output unit-to-pattern correspondance as there is in the categorical. Then, the reward carries more stochasticity than in the case of categorical policies.

In the categorical case, for an image size of $M = 1 6$ the output layer has a size of \~ 65k units given that the design space is $\{ 0 , 1 \} ^ { 1 6 \times 1 6 }$ . Larger images result challenging for a categorical parameterization, leaving Bernoulli kind of policies for these cases despite their requirement for longer training times. The stochastic reward assignment, instead of a direct pattern-todistribution reward, and the design space size are the trade-off variables for the framework for scaling up to large imaging problems.

## 4.5 Effect of sample-based approximations

We study sample-based approximations for state and reward, as introduced in Sections 3.3 and 3.4. This is important for translating this framework to more complex inference problems, where the posterior is accessible only through samples rather than in closed form. The focus of this section is to assess these effects separately, while in the Supplementary Data we present a brief study of both effects in combination with respect to the posterior without approximation.

## 4.5.1 State approximation

As described in Section 4.1, the posterior and RL state at time n is completely described by the discrete parameter $\mathbf { w } ^ { ( n ) }$ , whose entries $w _ { i } ^ { n }$ contains the probability that the active pixel is at location i. Therefore, the empirical mean state approximation computed from $N _ { s }$ posterior samples, as defined in (28), converges to the true state $\mathbf { w } ^ { ( n ) }$ . Here, we compare policies trained with this approximation against the exact posterior benchmark.

Figure 4 shows how approximating the state via sampling affects the learning process. Surprisingly, even the policy learned with approximating the state by a single sample reaches the optimal EIG. The inflection point around episode 6,000 marks the transition where modifying the number of samples produces a different learning performance. Strategies learned using fewer samples for the state approximation do need more episodes for reaching the optimal EIG. The CI95 plots show that the variability follows a similar behavior: Policies trained with a low number of samples need more episodes to reduce the variability, but eventually all of them to to zero.

## 4.5.2 Reward approximation

We train policies to recover the optimal one-active-pixel strategy using the sample-based EIG estimator introduced in Section 3.4 as the reward signal. This setting allows us to assess whether policies trained with the estimator in (32) can recover the strategy that is optimal under the exact EIG objective in (30). Although evaluating the exact EIG requires the marginal likelihood, which is generally intractable in more complex settings, it is available in the scenario we designed and provides a benchmark for evaluation.

In Figure 5 we observe that using the sample-based reward not only enables learning the optimal EIG strategy, it seems to even accelerate training.

![](images/8a05508f8c04b16a53ffc6e48cbe79cd58a6f6dc838ea26c3bfd7c900d68f5d0.jpg)

![](images/a6a6dfa0148fb51594257a5e2c8766d5edcf9575256d958e040b09b2711f8aa8.jpg)  
Figure 4: Total expected information gain for policy training under samplebased state approximations, with $N _ { s } = \{ 1 , 2 , 4 , 8 , 1 6 \}$ . First row figure displays mean total EIG, second row shows 95% confidence interval width. Curve for learning with no approximation, using exact state representation is illustrated in yellow for comparison of the sampling impact. Optimal policy reward is computed exactly and displayed in red for both plots.

The only case displaying a slower training curve than the exact reward is the $N _ { r } = 2$ case. In contrast to Figure 4, after the inflection point in episode 6,000, using more than $N _ { r } = 2$ samples for the reward approximation show a similar trend in mean total EIG and confidence interval widths to the case where the exact reward ('No Approximation') is used as a reward signal. Note that as in the sample-based state approximation (cf. Figure 4), the training curves converge to the exact reward curves for larger numbers of samples.

## 4.6 Learning under noisy measurements

In the noiseless case, the first measurement collapses the posterior to delta distributions over the regions of activity or inactivity in the case of measuring a 1 or 0. In the noisy case $\sigma > 0$ , this is no longer true, as noisy measurements still leave posterior probability mass over the discarded areas after the first acquisition. We now examine the impact of this on learning design policies. As a reference, we again compare their performance to the EIG of the bisection strategy, normalized as in (57).

Among the two approaches introduced in Section 3.4 to treat the Gaussian likelihood modeled in (1), our proposed one-active-pixel scenario requires the usage of a numerical integration method. For the experiments displayed in Figures 6 and 7, the rewards using the Bayesian quadrature are computed as described in Section 3.4, with $K = 1 0 0 0$ elements over the grid determined by each sample.

We can see that the learning behavior depends on the noise level: For a lower noise level, the policies learn slower than for the higher noise level at first but then overtake them and reach the reference normalized EIG earlier (note however, that the normalization depends on σ). For higher noise levels, they reach a sub-optimal inflection point in less episodes, as we observe for $\sigma = 0 . 8$ , but then it takes longer to converge to the reference EIG performance.

Learning an optimal EIG strategy directly at a fixed measurement noise level σ can be challenging, since noisy observations yield less informative posterior updates than in the noiseless setting. We investigate curriculum learning, in which training begins with noiseless measurements and the noise level is increased linearly to the target value σ over the course of training. The episode-dependent noise schedule is $\begin{array} { r } { \sigma _ { \mathrm { t r a i n } } ( i ) = \sigma \frac { i } { N _ { \mathrm { e p i s o d e s } } } } \end{array}$ , for $i = 0 , 1 , \ldots , N _ { \mathrm { e p i s o d e s } } .$ where i denotes the current episode. This curriculum is compared with the constant-noise training setting of Figure 6, in which $\sigma _ { \mathrm { t r a i n } } ( i ) = \sigma$ for every episode. To ensure a fair comparison, we evaluate both learned policies under the same target noise level σ and compute their total EIG exactly using (30).

Using $\sigma \in \{ 0 . 2 , 0 . 4 , 0 . 8 \}$ , Figure 7 displays a faster learning of the linear schedule after the inflection point is reached around episode 15,000 for all noise levels. After the first episodes, the 95% confidence intervals width shrink to zero for all $\sigma ,$ illustrating that using curriculum learning presents an performance advantage with respect to learning under constant noise. The linear curriculum learning policy is trained with cleaner measurement and evaluated in a noisier environment, which suggests that these strategies present a more stable training procedure for the proposed reinforcement learning framework.

![](images/07c89c1e01a4bac91efd35580b0e2ed086e2d1a1844333287d141246e0764eac.jpg)

![](images/d483b959948e9c4748e6bd434da3bf8fcb167cfec6ea25b119c80352b8dc9874.jpg)  
Figure 5: Total expected information gain using a sample-based reward approximation, with $N _ { r } = \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ samples and a exact state representation. Optimal rewards for each sample quantity are illustrated in red, with mean values in the first row plot and 95% confidence value widths over the second row. For comparison, EIG reward calculation using the posterior distribution representation without approximation ('No Approximation') is displayed in yellow.

![](images/651a79d979b3d0708d542d995c6ab73550f254f6195fea935cb1e583d32c2b60.jpg)

![](images/c9e41a775d8b5ab6b65bb39ed752f6e3e971a758e06cab6b4a1a43ff1a284563.jpg)  
Figure 6: Total normalized EIG for three noisy measurement levels $\sigma \in$ $\{ 0 . 2 , 0 . 4 , 0 . 8 \}$ . First column plot displays normalized EIG and second column plot 95% confidence interval widths, averaged across 100 categorical policies realizations. Curves are normalized using random and optimal baselines for each noise level, and represented in red and green, respectively.

![](images/3b9c25327a91ec5e2e53d79fbc73f3009b378ff386702bfd2f03c530b465d9e5.jpg)  
Figure 7: Total normalized EIG for three noisy measurement levels under constant and a linear curriculum learning for noise scheduling. Columns represent noise measurement levels $\sigma \in \{ 0 . 2 , 0 . 4 , 0 . 8 \}$ and rows display the normalized means and 95% confidence interval widths. Curves are normalized using random and optimal baselines for each noise level, and are drawn in gray and green lines, respectively,

## 4.7 Dose-constrained methods

In this study, we examine the two approaches to incorporate dose constraints introduced in Section 3.5. In the first case, the policy is a conditional Bernoulli distribution as in [28], where the design policy proposes patterns using a dose budget B along a experiment, as described in (40). Training is carried out for budgets $B \in \{ 1 , \ldots , 8 \}$ , the number of experiment steps is $M = 2$ and the image size is $N = 4$

Figure 8 shows the performance of constrained Bernoulli policies normalized by the optimal EIG and the strategy obtained using unconstrained

Bernoulli policies without training, i.e., projecting patterns at random without dose constraints, as described in Section 4.3 (this results in an average dose of 4 for this setup). We observe that only the trained policy with budget $B = 1$ cannot produce a policy that beats the unconstrained random policy, stabilizing at $\sim - 0 . 7 5$ of the normalized EIG. $B = 2$ policies start below the unconstrained Bernoulli random baseline but after training, these policies stabilize around a normalized EIG of $\sim 0 . 2 5$ with a large variability. This suggests that, even with a small budget, some of these policies produce dose-constrained strategies that are better than just drawing patterns from an unconstrained Bernoulli policy. Still, the budget is too small to reach optimal EIG strategies.

As we described in Section 4.2, an adaptive policy, i.e., one that truly implements sOED, needs a budget of $B = 3$ to reach optimal performance in this two-step experiment. We can see that the training curve indeed reaches an optimal EIG until episode 40,000 and then the performance drops in the last epochs. The policy presents the fastest learning of the optimal EIG strategy along with budget $B = 4$ , while this dose constraint is the first reaching almost zero variability around episode 25,000. Larger budget constraints $B \geq 5$ follow a similar trend, with a slower, more stable performance curve and low variability after the end of training.

Figure 9 illustrates how to incorporate the constraint as a penalty term, where λ is the weight between the dose budget B and the EIG to be optimized. We evaluate using $\lambda \in \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 4 , 0 . 0 8 , 0 . 1 , 0 . 2 , 0 . 4 \}$ and plot separately the EIG normalized between the optimal EIG strategy and the unconstrained random policy, the dose budget that each policy uses at the time of projecting patterns and the normalized 95% confidence interval.

When using the dose as a soft constraint, different EIG performances are observed by the end of the learning process. For small λ in the range of $[ 0 . 0 1 , 0 . 0 4 ]$ , the policies converge to an optimal EIG policy that proposes designs with $B = 4$ . Intermediate $\lambda \in [ 0 . 8 , 0 . 1 ]$ also display an optimal EIG behavior with a wider confidence interval and a mean budget $B \sim 3 . 5$ . In the range of large $\lambda \in \{ 0 . 2 , 0 . 4 \}$ , the dose constraint weights are so heavy that the resulting design no longer achieves optimal EIG. This constraint also proposes designs $B \sim 2$ that are suboptimal for the one-active pixel scenario and the confidence intervals are the widest.

## 5 Discussion

In Section 4 we illustrated the usage of the reinforcement learning framework to solve the Bayesian optimal experimental design of binary pattern for single-pixel imaging under a one-active pixel prior. For this methodology, we showed different aspects that affects the performance of the method and allow it to find optimal strategies incorporating constraints and measurement

![](images/347dac330fb07be16714b89da534bdf005ab7cd05001141bf04e6bbea6e45a7b.jpg)  
Figure 8: Total normalized EIG under dose budget constraints $B \in$ $\{ 1 , \ldots , 8 \}$ for conditional Bernoulli policies. First and second row display mean and 95% confidence interval width, respectively. Random baseline, in red, is computed using a unconstrained Bernoulli policy. Optimal baseline is drawn in green.

![](images/ac19786b7a9ce2751ca6dff82fbc5d1ea990b5c4d9452cc97d46b76e3a53aaec.jpg)

![](images/8364d7d76119943ea2239c2683476954ea3225d5912f9d257a9ab86c0d636780.jpg)

![](images/ae94d63ed035047b12b464f18edfdbd22f9dc1dc589c36a8365c626ae08e2d67.jpg)  
Figure 9: From top to bottom, total normalized EIG, dose budgets B and normalized 95% confidence interval widths. Dose constraints $\lambda \ \in$ $\{ 0 . 0 1 , 0 . 0 2 , 0 . 0 4 , 0 . 0 8 , 0 . 1 , 0 . 2 , 0 . 4 \}$ are used, with random and optimal baseline strategies in gray and red, respectively. For the dose budget plot, baselines represent a fixed (gray) or an adaptive (red) policy design choice.

noise.

When choosing a particular parameterization for the policy, the learning curves in Section 4.4 display an inflection point that replicates over the experiments. We infer that, given the two steps we set in the proposed SPI experiment, namely projecting two binary patterns to the image of interest, learning the optimal design splits into two phases. As we examined intermediate pattern proposals along training, this behavior is likely caused by learning a bisection halving pattern in the first place and later on learning the remaining strategy. This phenomena can be related to the adaptive and fixed strategies introduced in Section 4.2. At the first pattern, both strategies do not differ from each other, as they have a first bisection pattern with half of their pixels active. Moreover, the position of these active pixels is not crucial for the first pattern, but it is for the second one with respect to the first to achieve an optimal EIG step, since the input to the policy changed from being the prior in the first step to one of two possible posterior distributions in the second step. Then, the inflection point suggests a faster learning of the first pattern, where the policy training reaches a suboptimal EIG. Learning the second pattern takes more episodes for the policy due to the different possible posteriors as input for which a different optimal second pattern needs to be learned.

The sample-based state and reward studies in Section 4.5 show the potential of RL methods in large-scale imaging scenarios, where a Bayesian approach is only feasible via sampling methods. Here, we explored the effect of using state and reward estimators to learn optimal policies in the one-active pixel scenario. Our results show that training with a surprisingly small number of posterior samples is similarly fast compared to using the exact state or reward. The downside of this approach is the additional cost to simulating trajectories, which becomes more expensive in terms of compute due to the sampling times to compute the state or the reward estimators.

Lastly, training of policies under dose constraints in Section 4.7 demonstrate that better strategies than random Bernoulli patterns can be learned even for high dose constraints, i.e., low pattern activation budgets. These studies also directly demonstrate that our framework is able to learn truly adaptive policies, i.e., policies that solve the sOED problem (14), not just the a-priori OED problem (16): For both type of constraints, policies can only reach the optimal EIG with budget B = 3 if they act adaptively to the first measurement, cf. Section 4.2.

## 6 Conclusions

This work presents a way to formulate the design proposal of single-pixel imaging binary patterns as a Bayesian sequential optimal experimental design problem. By using a reinforcement learning framework, we proposed different ways to parameterize a design policy, and illustrated how they can be used depending on the scale of the design size and the image size. For random binary patterns, if the image size is small, a categorical policy over all possible options is preferred over a Bernoulli one due to their faster training procedure.

To investigate the scaling from proof-of-concept to large scale computational imaging scenarios, we carried out sample-based approximations using numerical simulations. Surprisingly, we found that approximating the state and/or the reward with with sample-based estimators had only minor effects on the learning dynamics, even for very low numbers of samples.

We validated the robustness of the framework by learning designs when the measurements contain different noise levels. To accelerate training in this scenario, we showed that a curriculum learning strategy can be beneficial that uses a gradual noise scaling over the training episodes, simulating a degree of difficulty from simple to complex.

Moreover, we studied two possible ways to impose constraints over the illumination dosage, enabling budget-constrained single-pixel experimental designs and proofing that the learned designs are truly adaptive.

In this work, we deliberately limited our numerical studies to a simple small-scale scenario, for which we could deduce optimal policies in an intuitive way. This allowed us to evaluate the learned policies in a thorough way and compute many quantities that are usually not tractable. In future work, we will extend this framework to other image priors and large-scale SPI scenarios.

## Data availability

The code supporting the findings of this study will be released publicly upon acceptance of the manuscript.

## Ethics statement

This study did not involve research with human participants, human data, human tissue, or live animals. Therefore, ethical approval was not required.

## Funding

The authors have no conflicts of interest to declare. This work was supported by the European Union (No. GA 101072354). Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## Acknowledgements

M.A.O. thanks L. Veenstra for insightful discussions on Bayesian inverse problems.

## References

[1] M. F. Duarte, M. A. Davenport, D. Takhar, J. N. Laska, T. Sun, K. F. Kelly, and R. G. Baraniuk, "Single-pixel imaging via compressive sampling," IEEE Signal Processing Magazine, vol. 25, no. 2, pp. 83–91, 2008.

[2] Z. Zhang, X. Wang, G. Zheng, and J. Zhong, "Hadamard single-pixel imaging versus fourier single-pixel imaging," Optics Express, vol. 25, no. 16, pp. 19 619–19 639, 2017.

[3] R. Stojek, A. Pastuszczak, P. Wróbel, and R. Kotyński, "Single pixel imaging at high pixel resolutions," Optics express, vol. 30, no. 13, pp. 22 730–22 745, 2022.

[4] D. L. Donoho, "Compressed sensing," IEEE Transactions on Information Theory, vol. 52, no. 4, pp. 1289–1306, 2006.

[5] E. J. Candès, J. Romberg, and T. Tao, "Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information," IEEE Transactions on Information Theory, vol. 52, no. 2, pp. 489–509, 2006.

[6] L. Gan, T. T. Do, and T. D. Tran, "Fast compressive imaging using scrambled block hadamard ensemble," in 2008 16th European Signal Processing Conference. IEEE, 2008, pp. 1–5.

[7] T. Rainforth, A. Foster, D. R. Ivanova, and F. Bickford Smith, "Modern bayesian experimental design," Statistical Science, vol. 39, no. 1, pp. 100–114, 2024.

[8] W. Shen and X. Huan, “Bayesian sequential optimal experimental design for nonlinear models using policy gradient reinforcement learning," Computer Methods in Applied Mechanics and Engineering, vol. 416, p. 116304, 2023.

[9] W. J. Welch, “Branch-and-bound search for experimental designs based on D optimality and other criteria," Technometrics, vol. 24, no. 1, pp. 41–48, 1982.

[10] A. Foster, D. R. Ivanova, I. Malik, and T. Rainforth, "Deep adaptive design: Amortizing sequential Bayesian experimental design," in Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021, pp. 3384–3395.

[11] O. Scherzer, M. Grasmair, H. Grossauer, M. Haltmeier, and F. Lenzen, Variational Methods in Imaging, ser. Applied Mathematical Sciences. New York, NY: Springer, 2009, vol. 167.

[12] A. M. Stuart, “"Inverse problems: A bayesian perspective," Acta Numerica, vol. 19, pp. 451–559, 2010.

[13] M. Lassas and S. Siltanen, “Can one use total variation prior for edgepreserving bayesian inversion?" Inverse problems, vol. 20, no. 5, p. 1537, 2004.

[14] G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher, “An analysis of approximations for maximizing submodular set functions," Mathematical Programming, vol. 14, no. 1, pp. 265–294, 1978.

[15] A. Alexanderian and S. Maio, "Submodularity of the expected information gain in infinite-dimensional linear inverse problems," arXiv preprint arXiv:2602.09285, 2026.

[16] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. MIT press, 2018.

[17] R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning," Machine learning, vol. 8, pp. 229–256, 1992.

[18] R. S. Sutton, D. A. McAllester, S. P. Singh, and Y. Mansour, "Policy gradient methods for reinforcement learning with function approximation," in Advances in Neural Information Processing Systems, vol. 12, 2000.

[19] B. T. Vincent and T. Rainforth, "The darc toolbox: automated, flexible, and efficient delayed and risky choice experiments using bayesian adaptive design," PsyArXiv preprint, vol. 5, p. 9, 2017.

[20] X. Huan and Y. M. Marzouk, "Simulation-based optimal Bayesian experimental design for nonlinear systems," Journal of Computational Physics, vol. 232, no. 1, pp. 288–317, 2013.

[21] R. A. Horn and C. R. Johnson, Matrix Analysis, 2nd ed. Cambridge: Cambridge University Press, 2013.

[22] K. Chaloner and I. Verdinelli, “Bayesian experimental design: A review," Statistical Science, vol. 10, no. 3, pp. 273–304, 1995.

[23] V. V. Fedorov, Theory of Optimal Experiments. New York: Academic Press, 1972.

[24] P. P. Laissue, R. A. Alghamdi, P. Tomancak, E. G. Reynaud, and H. Shroff, “Assessing phototoxicity in live fluorescence imaging," Nature Methods, vol. 14, no. 7, pp. 657–661, 2017.

[25] J. Icha, M. Weber, J. C. Waters, and C. Norden, "Phototoxicity in live fluorescence microscopy, and how to avoid it," BioEssays, vol. 39, no. 8, p. 1700003, 2017.

[26] A. Attia, "Probabilistic approach to black-box binary optimization with budget constraints: Application to sensor placement," arXiv preprint arXiv:2406.05830, 2024.

[27] R. S. Sutton and A. G. Barto, “Temporal-difference learning," Reinforcement learning: an introduction, pp. 133–160, 1998.

[28] A. Attia and E. Constantinescu, “Optimal experimental design for inverse problems in the presence of observation correlations," SIAM Journal on Scientific Computing, vol. 44, no. 4, pp. A2808–A2842, 2022.

## Supplementary data

Section 4.5 introduced separate studies varying the number of samples over the state and reward computation. In Figure 10 we study the effect of reusing the same samples for approximating the state and the reward. In comparison to the previous approximation studies, we observe a similar behavior to the reward approximation for $N _ { s } , N _ { r } \ge 4$ , where the total EIG along training is similar with no different learning trend after the inflection point at episode 6,000. For $N _ { s } , N _ { r } ~ = ~ 2$ , the learning curve is slower and the number of episodes is not enough for reaching the optimal EIG. With respect to the reward approximation alone, using $N _ { s } , N _ { r } = 2$ seems to have a stronger impact in learning the optimal strategy, suggesting that there is a strong interplay between state and reward approximation when samples are scarce.

![](images/b34f15b0f8910db99ae74647e5a45a56d9d11a3596614b58be8124e369408491.jpg)  
Figure 10: Impact of reward approximation on policy learning. Training curves showing normalized cumulative EIG for categorical policy learning under a Monte Carlo reward approximation with $N _ { s } , N _ { r } \in$ {2, 4, 8, 16, 32} posterior samples. The same samples are used to compute both state and reward approximations. Curves are averaged across 100 independent policy training realizations.