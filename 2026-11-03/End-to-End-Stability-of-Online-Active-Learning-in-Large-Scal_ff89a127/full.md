# End-to-End Stability of Online Active Learning in Large-Scale Neural Systems

Shuiwen Shen<sup>1,7,\*</sup>, Jinhua Zhang<sup>2</sup>, Chenlei Hou<sup>3</sup>, Peng Shi<sup>4</sup>, Yong Han<sup>1</sup>, Bo Gao<sup>5</sup>, Xiaoxiao Na<sup>6</sup>, Zhihui Lin<sup>1</sup>, Pengtao Du<sup>1</sup>

<sup>1</sup>School of Mechanical and Automotive Engineering, Xiamen University of Technology, Xiamen, 361024, China.

<sup>2</sup>China Innovation Center for Intelligent and Connected Vehicles, Beijing, 100176, China.

<sup>3</sup>King Long United Automotive Industry Co. Ltd., Xiamen, Fujian, 361023, China.

<sup>4</sup>School of Electrical and Mechanical Engineering, Adelaide University, Adelaide, SA 5005, Australia.

<sup>5</sup>Syselek UK Ltd, Warwick, CV34 6RG, United Kingdom.

<sup>6</sup>Department of Engineering, Cambridge University, Cambridge, CB2 1TN, United Kingdom.

<sup>7</sup>State Innovative Centre for Automotive Electric Driving at Xiamen, Xiamen, Fujian, 361010, China.   
2023000085@xmut.edu.cn.

## Abstract

Online Active Learning (OAL) aims to extend deep learning from static inference after ofline training to continuous adaptation during real-time interaction with a changing physical environment. This transition is essential for safety critical applications, but it raises a central theoretical question: whether the online parameter update process remains stable when scaled to multilayer neural architectures. This paper develops a stability focused OAL framework. The framework uses prediction error as a feedback signal to update a structured deep model, thereby linking online learning with Lyapunov based energy minimization. We show that prediction error convergence alone is insuficient for global parameter convergence because multilayer networks may admit high dimensional manifolds of equivalent output solutions. The analysis therefore separates output error regulation, internal zero dynamics, interlayer coupling, and parameter convergence. Persistency of excitation and active damping are then used to remove the residual null space modes that remain after output error convergence. A cascaded Lyapunov analysis establishes asymptotic stability of the overall OAL learning process, while monotonic activation functions and difeomorphic layer mappings provide the uniqueness condition for the target parameter trajectory. The resulting proof provides a theoretical basis for applying online adaptive learning to large scale neural systems in nonstationary and safety critical settings.

Keywords: Online Active Learning, Stability analysis, Lyapunov stability, Adaptive learning, Structural isomorphism, Persistency of excitation, Active damping, Safety-critical AI

## 1 Introduction

Online learning is essential for applications operating in non-stationary physical environments. In contrast to ofline-trained deep learning models, whose parameters are typically fixed after deployment, an online learning system must continuously update its internal model according to real-time sensory feedback. This capability is particularly important for safety-critical systems, where distributional shifts, unmodeled disturbances, and out-of-distribution inputs can cause prediction errors to accumulate during operation. However, continuous parameter adaptation also introduces a fundamental theoretical challenge: the learning process itself must remain stable.

Existing deep learning architectures are highly scalable, but their stability is usually established within an ofline optimization setting. Backpropagation provides an efective mechanism for minimizing empirical loss over a fixed dataset, yet it does not directly guarantee stable real-time adaptation when data arrive sequentially from a changing physical environment [1, 2]. Concerns about global credit assignment and biological plausibility have consequently motivated more local learning alternatives, including the Forward–Forward algorithm [3]. More broadly, biological intelligence relies on inherited structure and recurrent organization that are not captured by pure learning alone [4]. Continual-learning and test time-adaptation methods likewise seek to update deployed models under distribution shift [5–7], but their optimization objectives do not by themselves provide an end-to-end Lyapunov stability guarantee.

Classical adaptive learning and control theory, by contrast, provide mature tools for proving convergence and stability through Lyapunov analysis, but these methods are often limited by the dificulty of representing large-scale nonlinear systems with high-dimensional parameter spaces. This creates a gap between the scalability of deep learning and the formal stability requirements of online adaptive systems.

Online Active Learning (OAL) is introduced to bridge this gap. The central idea of OAL is to use prediction error as a continuous feedback signal for updating a structured deep model in real time. Rather than treating learning as a separate ofline training phase, OAL embeds learning into the system’s closed-loop interaction with the external world. The predictive model receives sensory input, generates an internal prediction, evaluates the discrepancy between prediction and observation, and updates its parameters through an error-feedback learning law. In this sense, OAL transforms deep learning from a static inference architecture into a dynamic adaptive system.

The key question addressed in this paper is whether such an online learning process can be made stable at large scale. A naive error-feedback update can reduce the output prediction error, but it does not automatically guarantee convergence of the internal parameters. In multilayer neural systems, diferent parameter configurations may produce the same output, leaving the learning trajectory on a high-dimensional manifold of equivalent solutions. As a result, prediction-error convergence alone is insuficient for proving global learning stability. A stability proof must therefore account not only for the decay of output error, but also for parameter convergence, cross-layer coupling, and internal zero dynamics.

This paper develops a stability-centered analysis of OAL. Section 2 formulates the primary OAL framework and its error-feedback learning law. Section 3 then analyzes why non-increasing predictionerror energy does not by itself imply parameter convergence. The proof separates the marginal zero dynamics from the damped cascaded dynamics, identifies the excitation and damping conditions needed to remove the null-space ambiguity, and derives the resulting global Lyapunov stability condition.

## 2 Principle of Online Active Learning

## 2.1 Deep learning and adaptive learning foundations

Deep learning and adaptive learning provide complementary foundations for online model adaptation. Deep learning ofers a scalable representation mechanism for high-dimensional nonlinear mappings, while adaptive learning provides a feedback-based view of how a model can change during operation. In a conventional deep neural network, the predictive output can be written abstractly as

$$
\hat { \mathbf { y } } = f \mathbb { N } \mathbb { N } _ { \mathrm { d l } } ( \mathbf { x } , \mathbf { \boldsymbol { \theta } } ) ,\tag{1}
$$

where x denotes the input, $\hat { \mathbf { y } }$ the predicted output, and θ the trainable parameters. The corresponding layer-wise architecture is denoted by

$$
\mathbf { z } ^ { [ i ] } = \boldsymbol { \Theta } ^ { [ i ] } \mathbf { y } ^ { [ i - 1 ] } , \qquad \mathbf { y } ^ { [ i ] } = g _ { i } ( \mathbf { z } ^ { [ i ] } ) , \qquad i = 1 , 2 , \ldots , n ,\tag{2}
$$

where $g _ { i }$ is the nonlinear activation function of layer i. This linear-layer-plus-activation structure gives deep learning its representational capacity [8] and preserves a repeated layer-wise form across the network. Successive layers can also be interpreted as progressively transforming or compressing the information needed for the downstream task [9]. However, in the standard ofline paradigm, the parameters are optimized before deployment and are then held fixed during inference.

The ofline optimization process is commonly driven by backpropagation. If L denotes the training loss, the layer-wise gradient recursion can be summarized as

$$
\begin{array} { c } { \displaystyle \mathbf { g } _ { i } = \frac { \partial \mathbf { y } ^ { [ i + 1 ] } } { \partial \mathbf { y } ^ { [ i ] } } \mathbf { g } _ { i + 1 } , } \\ { \displaystyle \mathbf { g } _ { \mathbf { \theta } _ { i } } = \frac { \partial \mathbf { y } ^ { [ i ] } } { \partial \mathbf { \theta } ^ { [ i ] } } \mathbf { g } _ { i } , } \\ { \Delta \mathbf { \Theta } ^ { [ i ] } = - \lambda \mathbf { g } _ { \mathbf { \theta } _ { i } } . } \end{array}\tag{3}
$$

Here $\mathbf { g } _ { i }$ and $\mathbf { g } _ { \boldsymbol { \Theta } _ { i } }$ are the gradients associated with $\mathbf { y } ^ { [ i ] }$ and $\boldsymbol { \Theta } ^ { [ i ] }$ , respectively. This recursive structure is one reason deep networks can remain trainable at very large parameter scales. It also provides the structural template later used to interpret online feedback learning as a layer-wise isomorphic process. However, this mechanism is designed for learning from a fixed dataset and does not by itself define a stable online adaptive system. Once the external environment changes, the model may accumulate prediction error without any principled way to update its parameters during operation.

Adaptive learning starts from the opposite perspective. Rooted in adaptive-control theory [10, 11], it represents the process by a parameterized state-space model and treats prediction error as a real-time feedback signal. For an adaptive model with estimated state $\hat { \mathbf { s } } ,$ input x, and predicted output $\hat { \mathbf { y } }$ , the model can be described abstractly as

$$
\dot { \hat { \mathbf { s } } } = f ( \hat { \mathbf { s } } , \theta , \mathbf { x } ) , \qquad \hat { \mathbf { y } } = g ( \hat { \mathbf { s } } , \theta ) .\tag{4}
$$

The tracking error is then defined as

$$
\begin{array} { r } { \mathbf { e } _ { y } = \mathbf { y } _ { s } - \hat { \mathbf { y } } _ { s } , } \end{array}\tag{5}
$$

where ${ \bf y } _ { s }$ and $\hat { \textbf { y } } _ { s }$ are the measured and predicted sensory outputs, respectively. Adaptive learning updates the model according to this feedback. In a generic form, the learning dynamics can be written as

$$
\begin{array} { r } { \dot { \bf \theta } = \lambda h ( { \bf e } _ { y } , \hat { \bf s } ) , } \end{array}\tag{6}
$$

where λ is an adaptation gain. If the mapping from $\boldsymbol { \Theta } \cdot \hat { \mathbf { s } }$ to $\mathbf { e } _ { y }$ is Strictly Positive Real (SPR), the following simplified learning law can guarantee stability and solution uniqueness [12]:

$$
\dot { \boldsymbol { \Theta } } = \lambda \mathbf { e } _ { y } \hat { \mathbf { s } } .\tag{7}
$$

This feedback view is attractive for online operation, but the SPR condition is often violated in large nonlinear systems, particularly in the presence of unmodeled dynamics. Moreover, the solution to ${ \bf e } _ { y } =$ 0 is not necessarily unique in high-dimensional parameter spaces. Classical adaptive formulations are therefore dificult to scale directly to deep models with many layers and many parameters [10, 11].

## 2.2 Primary OAL framework and error-feedback learning law

Online Active Learning (OAL) combines the representational capacity of deep learning with the feedback structure of adaptive learning. In its conceptual form, a single deep model as in Eqn (1) is corrected online by the sensory prediction error, as illustrated in Fig. 1a, where the model is driven by an external stimulus x while the sensor measurement y serves as the comparison target. This view keeps the layerwise error propagation structure of backpropagation [1, 2, 13], but replaces the ofline loss gradient with an online error signal generated by the mismatch between physical-world measurements and model predictions. It is also consistent with the hierarchical error-propagation idea in predictive coding [14–17] and with the Bayesian view that perception continuously reconciles internal predictions with uncertain sensory evidence [18]. Related active-inference models similarly couple prediction and learning through generative state-space dynamics [19]. Unlike the cited formulations, which emphasize inference over latent states, the present analysis focuses on Lyapunov stability of the parameter-adaptation dynamics in a large-scale layered model.

![](images/23ea23f2916d209967af9f52138236196d46e9fac89df1bd204b40aafd110bf9.jpg)  
(a) Principal framework

(b) Primary framework  
![](images/fe7ec516ce5c0e1cdcc6d26185b1ac1fb16df41f9675c990e106ba5a81afc73f.jpg)  
Fig. 1 From the principal Online Active Learning framework to the primary framework. (a) The external stimulus x drives both the true physical process and the structured deep model, while the sensor measurement y is the comparison target. The Physical World block denotes the true process, whereas Eqn (4) describes its parameterized internal model. The gradient arrows correspond to $\mathbf { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \mathbf { e } ^ { [ i ] }$ in Eqn (11), up to the gradient-sign convention. (b) The primary framework $\mathbb { F } _ { \mathrm { O A L } } ^ { \mathrm { P R Y } }$ sets $\mathbf { x } = \mathbf { y } _ { s }$ and decomposes the model into prediction P and reconstruction R, with the resulting error $\mathbf { e } _ { y }$ distributed across the layers for online parameter correction.

The conceptual framework is already suficient to drive the predicted output towards the measurement, but it leaves two questions open. The error-driven correction alone provides no mechanism for extracting a compact internal representation of the external world, and it gives no guarantee that the model parameters themselves converge. From a system-dynamics perspective the second question is the more fundamental one, because the same output can be produced by many diferent parameter configurations. The first question is answered by restructuring the model, as described below; the second is the subject of Section 3.

To obtain a representation, the monolithic model of Eqn (1) is decomposed into a prediction model P, which compresses the sensory input into a low-dimensional latent space, an optional latent model L, which evolves the latent state, and a reconstruction model R, which maps the latent state back to the observation space. Predictive tasks are then carried out in the latent space, while the reconstruction supplies the learning signal. Setting the model input directly to the sensory stream, $\mathbf { x } = \mathbf { y } _ { s }$ , yields the primary framework $\bar { \mathbb { F } } _ { \mathrm { O A L } } ^ { \mathrm { P R Y } }$ shown in Fig. 1b. The system then interacts with the external world exclusively through its sensory boundary, in the manner of a Markov blanket [20], and can learn from a continuously arriving stream without an externally supplied label.

The framework operates as a repeated prediction–evaluation–correction cycle. During prediction, $\textbf { y } _ { s }$ passes through the prediction model P to form the latent representation $\hat { \mathbf { y } } _ { z }$ , after which the reconstruction model R produces $\hat { \textbf { y } } _ { s }$ . Both models are layered neural networks of the form defined in Eqn (2), and their complete forward mapping is represented by Eqn (1). In the primary diagram, any latent dynamics model L is implicit and may be absorbed into P or R. It is written explicitly later only to express the general learning dynamics used in the stability analysis. The latent representation $\hat { \mathbf { y } } _ { z }$ is also the framework output marked as Predictions in Fig. 1b. It is consumed by the downstream task, whereas reconstruction is used to generate the online learning signal. For example, in a visual perception system, P may compress the incoming camera or sensor stream into a latent description of lane geometry, obstacles, or motion context. The reconstruction branch then tests whether this latent state still explains the current sensory signal. The reconstruction error provides the online correction signal, while the latent prediction remain available for planning or control.

During evaluation, the reconstructed signal is compared with the current sensory measurement according to Eqn (5). During correction, this same error is propagated backward from R to P and updates every constituent layer. The corrected model then processes the next measurement, so inference and learning proceed concurrently. This closed loop provides the organizing structure for all equations below.

Because the reconstruction target is the input itself, the primary framework admits the degenerate solution $\mathbb { R } = \mathbb { P } ^ { - 1 }$ , which drives the error to zero without capturing any physical structure of the environment. The practical spatial and temporal variants of OAL break this symmetry by reconstructing a diferent sensory channel or the next frame in time. Both retain the encoder–latent–decoder structure and the error-feedback learning law of Fig. 1b, so the stability analysis below applies to them without modification.

The basic energy quantity minimized by OAL is the prediction-error energy

$$
\mathcal { L } _ { e } = \frac { 1 } { 2 } \mathbf { e } _ { y } ^ { T } \mathbf { e } _ { y } ,\tag{8}
$$

where $\mathbf { e } _ { y }$ is the sensory prediction error generated at the comparison node in Fig. 1b. Inside the Online Active Learning block, this error is propagated backward and converted into $\mathbf { e } ^ { [ 1 ] } , \ldots , \mathbf { e } ^ { [ n ] }$ . Each layer receives its corresponding error signal for parameter adaptation. The local error of layer i is written as

$$
\mathbf { e } ^ { [ i ] } = \mathbf { J } _ { y } ^ { [ i , i + 1 ] } \mathbf { e } ^ { [ i + 1 ] } ,\tag{9}
$$

where $\mathbf { J } _ { y } ^ { [ i , i + 1 ] }$ denotes the backward output sensitivity between adjacent layers. The corresponding parameter sensitivity is

$$
{ \bf J } _ { \boldsymbol { \Theta } } ^ { [ i ] } = \frac { \partial { \bf y } ^ { [ i ] } } { \partial { \bf \boldsymbol { \Theta } } ^ { [ i ] } } ,\tag{10}
$$

and the local online update law is summarized as

$$
\Delta \mathbf { \boldsymbol { \Theta } } ^ { [ i ] } = \lambda _ { i } \mathbf { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \mathbf { e } ^ { [ i ] } .\tag{11}
$$

For brevity, $\mathbf { J } _ { y } ^ { [ i , i + 1 ] }$ is written as $\mathbf { J } _ { y } ^ { [ i ] }$ , with $\mathbf { J } _ { y } ^ { [ n ] } = \mathbf { I }$ . The global sensitivities along the backward path are

$$
\mathbb { J } _ { y } ^ { [ i ] } = \prod _ { k = i } ^ { n } \mathbf { J } _ { y } ^ { [ k ] } .\tag{12}
$$

The corresponding global parameter sensitivity is

$$
\mathbb { J } _ { \boldsymbol { \theta } } ^ { [ i ] } = \mathbf { J } _ { \boldsymbol { \theta } } ^ { [ i ] } \mathbb { J } _ { \boldsymbol { y } } ^ { [ i ] } ,\tag{13}
$$

so that the learning law can also be written in global form as

$$
\Delta \Theta ^ { [ i ] } = \lambda _ { i } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \mathbf { e } _ { y } .\tag{14}
$$

This global form shows how the single output error in Fig. 1b reaches every layer of R and P. The model changes online because each layer receives an error-driven correction, but the stability of the resulting coupled multilayer dynamics still has to be proven.

At the prediction-error level, the OAL update law leads to a Lyapunov-like descent condition. Under bounded external variation and positive learning gain, the derivative of the prediction-error energy can be expressed in the simplified form

$$
\begin{array} { r } { \dot { \mathcal { L } } _ { e } = - \lambda \mathbf { e } _ { y } ^ { T } \mathbb { J } _ { \theta } ^ { T } \mathbb { J } _ { \theta } \mathbf { e } _ { y } \le 0 . } \end{array}\tag{15}
$$

A computationally relaxed update may also use the sign of the sensitivity matrix,

$$
\Delta \Theta ^ { [ i ] } = \lambda _ { i } \mathrm { s g n } ( \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } ) { \bf e } ^ { [ i ] } ,\tag{16}
$$

because sgn $( \mathbb { J } _ { \theta } ) ^ { T } \mathbb { J } _ { \theta } \geq 0$ preserves the non-negative sensitivity product required by the stability argument while avoiding full Jacobian evaluation at every step.

For the stability proof, the latent model L that is implicit in Fig. 1b is written out explicitly. Propagating the prediction error through $\mathbb { R }  \mathbb { L }  \mathbb { P }$ gives the damped learning laws

$$
\begin{array} { r } { \dot { \bf { \theta } } _ { R } + \kappa _ { R } { \bf { \theta } } _ { R } = \lambda _ { R } { \bf { J } } _ { \theta _ { R } } ^ { T } { \bf { e } } _ { R } , } \end{array}
$$

$$
\mathbf { e } _ { R } = \mathbf { e } _ { y } ,\tag{17}
$$

$$
\dot { \bf \theta } _ { L } + \kappa _ { L } \pmb { \theta } _ { L } = \lambda _ { L } \mathbf { J } _ { \theta _ { L } } \mathbf { e } _ { L } ,
$$

$$
\begin{array} { r } { \mathbf { e } _ { L } = \mathbf { J } _ { L } \mathbf { e } _ { R } , } \end{array}\tag{18}
$$

$$
\begin{array} { r } { \dot { \bf { \theta } } _ { P } + \kappa _ { P } { \bf { \theta } } _ { P } = \lambda _ { P } { \bf { J } } _ { \theta _ { P } } { \bf { e } } _ { P } , } \end{array}
$$

$$
\mathbf { e } _ { P } = \mathbf { J } _ { P } \mathbf { e } _ { L } ,\tag{19}
$$

where the learning inertia coeficient $\kappa _ { ( \cdot ) } > 0$ originates from parameter regularization and acts as a damping term, written layer-wise as $\mathbb { K } _ { \boldsymbol { \theta } } ^ { [ i ] }$ in Section 3, that preserves internal priors and enhances real-time learning stability. These equations establish the principle that OAL learning consists of feedback-driven adaptive parameter loops with explicit damping. The following section therefore focuses on the central theoretical question: whether these coupled loops remain globally stable when extended to large-scale multilayer neural systems.

## 3 Stability Proof of Online Active Learning

## 3.1 End-to-end global learning stability

As shown in Subsection 2.2, the OAL error-feedback law yields a non-increasing prediction-error energy under the stated bounded-variation and positive-gain assumptions. This property provides the basic online error-regulation mechanism for AI models. However, non-increasing output error energy alone does not guarantee the convergence of self-adapting parameters, which is addressed in this section.

## Root of the problem with global convergence

Consider the parameters θ of the deep learning model defined in Eqn(1), represented by the tensor

$$
\pmb { \theta } = [ \pmb { \theta } ^ { [ 1 ] } , \pmb { \theta } ^ { [ 2 ] } , \cdots , \pmb { \theta } ^ { [ i ] } , \cdot \cdot \cdot \pmb { \theta } ^ { [ n ] } ] ^ { T }
$$

where $\boldsymbol { \Theta } ^ { [ i ] }$ denotes the parameters for the i-th layer of the deep neural network model spanning the prediction, latent, and reconstruction models P, L, and R, respectively. This formulation generalizes to Convolutional Neural Networks (CNNs) and Transformer architectures, which can be mathematically treated as sparse variants of $\boldsymbol { \Theta } ^ { [ i ] }$ , in general. Matrix $\boldsymbol { \Theta } ^ { [ i ] }$ is decomposed as

$$
\pmb { \theta } ^ { [ i ] } = [ \pmb { \theta } _ { 1 } ^ { [ i ] } , \pmb { \theta } _ { 2 } ^ { [ i ] } , \cdots , \pmb { \theta } _ { k } ^ { [ i ] } , \cdots \pmb { \theta } _ { n _ { i } } ^ { [ i ] } ] ^ { T }\tag{20}
$$

in which $n _ { i }$ indicates the width of the corresponding layer, and $\boldsymbol { \Theta } _ { k } ^ { [ i ] }$ the weight vector for the k-th node of i-th layer. Performing a first order Taylor expansion of the predictive error $\mathbf { e } _ { y }$ as in Eqn(5) with respect

to parameter θ, considering Eqn(1), Eqn(9), and Eqn(10) yields

$$
\mathbf { e } _ { y } = \mathbb { J } _ { \boldsymbol { \Theta } } ^ { T } \delta \boldsymbol { \Theta } = \Sigma _ { i } ( \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } ) ^ { T } \delta \boldsymbol { \Theta } ^ { [ i ] } = \Sigma _ { i } \Sigma _ { k } ( \mathbb { J } _ { \boldsymbol { \Theta } _ { k } } ^ { [ i ] } ) ^ { T } \delta \boldsymbol { \Theta } _ { k } ^ { [ i ] }\tag{21}
$$

or simply as

$$
\mathbf { e } _ { y } = \mathbb { J } _ { \boldsymbol { \theta } _ { k } ^ { [ i ] } } ^ { T } \delta \boldsymbol { \Theta } _ { k } ^ { [ i ] }
$$

according to Einstein notation for tensor calculation. In which

$$
\delta \boldsymbol { \Theta } = \boldsymbol { \Theta } ^ { * } - \boldsymbol { \Theta }\tag{22}
$$

represents the deviation<sup>1</sup> of the current parameter from the ideal targets $\theta ^ { * }$ which captures the external world perfectly. Equivalent definitions apply to $\delta \Theta ^ { [ i ] }$ and $\delta \Theta _ { k } ^ { [ i ] }$ . Because the parameter adaptation process is driven directly by the error feedback flow of $\mathbf { e } _ { y }$ according to Eqn(14), no further parameter update is generated in principle if ${ \bf e } _ { y } = 0$ . This equilibrium state follows from the minimum-energy law in Eqn(15). Under these conditions, Eqn(21) reduces $\mathrm { { t o } ^ { 2 } }$

$$
\underline { { \mathbb { J } } } _ { \boldsymbol { \Theta } } ^ { [ 1 ] ^ { T } } \delta \boldsymbol { \Theta } ^ { [ 1 ] } + \underline { { \mathbb { J } } } _ { \boldsymbol { \Theta } } ^ { [ 2 ] ^ { T } } \delta \boldsymbol { \Theta } ^ { [ 2 ] } + \cdots + \underline { { \mathbb { J } } } _ { \boldsymbol { \Theta } } ^ { [ i ] ^ { T } } \delta \boldsymbol { \Theta } ^ { [ i ] } + \cdots + \underline { { \mathbb { J } } } _ { \boldsymbol { \Theta } } ^ { [ n ] ^ { T } } \delta \boldsymbol { \Theta } ^ { [ n ] } = \mathbf { 0 }\tag{23}
$$

This demonstrates that the error feedback update law alone is insuficient to ensure global convergence toward $\delta \mathbf { \boldsymbol { \Theta } } = \mathbf { \boldsymbol { 0 } } .$ , both in terms of solution uniqueness and stability. Instead, it defines a high-dimensional manifold<sup>3</sup> on which the resulting parameters may undergo stochastic sliding. Crucially, this equation indicates that the global parameter learning processes are strongly coupled through linear dependence.

![](images/a82588e15a32520f860556f90c4f31ba8980fb413a2fa1023dc55edd4e633e85.jpg)  
Computational Fluid Dynamics

![](images/dac467e7005395cd7786e892230e170efe9c9fe5872985c6720c5953810a4be3.jpg)  
Finite Element Method

![](images/26a162e9e628cb88770f9bcb35fa86dc96b7babdce05ec6a712a8d6122836196.jpg)  
Thermal Dynamics  
Fig. 2 Structural Isomorphism of Large-Scale System Dynamics in Engineering

However, the structural isomorphism of the feedback learning mechanism as given by Eqn(3) implies that the same layer-wise argument can be applied across the parallel processing pathways. This observation does not by itself collapse the stationary manifold in Eqn(23). Rather, it provides the structural template used below to analyze why additional excitation or damping is required to remove the residual zero dynamics.

The stability analysis is developed through an iterative deduction based on the structural isomorphism of the layered learning dynamics. The stability of such massive dynamical systems is also supported by empirical and physical analogies. In engineering, this is exemplified (see Fig.2) by Computational Fluid Dynamics (CFD), the Finite Element Method (FEM), and thermodynamics. These disciplines successfully capture huge-scale system dynamics precisely because the constituent elements share an identical structural framework.

This principle of underlying structural isomorphism mirrors fundamental processes in the natural world. For instance, macroscopic fluid mechanics and thermodynamics emerge from repeated local interactions among large numbers of individual molecules. Similarly, the macroscopic behavior governed by solid mechanics arises from collective interactions across vast crystalline lattices. These engineering paradigms and physical phenomena motivate the use of repeated local structure as an analytical simplification for ultra-large-scale dynamics.

## Isomorphism of layered feedback learning

The feedback learning framework exhibits an identical learning topology across layers despite variations in latent-space dimensionality. Figure 3 visualizes this topological isomorphism and supports the layer-wise analysis below.

• Layered Parallel Learning. Each individual layer undergoes adaptive optimization driven by its respective layer-wise error $\mathbf { e } ^ { [ i ] }$ , according to Eqn(11). For the stability analysis, the governing learning law is reformulated here in dynamic form.

$$
\dot { \boldsymbol { \Theta } } ^ { [ i ] } = \lambda _ { i } \mathbf { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \mathbf { e } ^ { [ i ] }\tag{24}
$$

This law remains topologically and structurally invariant across all layers. Notably, the underlying manifold possesses high dimensionality, given that θ<sup>[i]</sup> scales as an $n _ { i } \times n _ { i - 1 }$ matrix.

• Learning Strategy Symmetry. The optimization protocol displays strict layer-wise symmetry. The learning strategies for layer i, and $j$ are as, respectively

$$
{ \dot { \boldsymbol { \Theta } } } ^ { [ i ] } = \lambda _ { i } \mathbf { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \mathbf { e } ^ { [ i ] } ; \quad { \dot { \boldsymbol { \Theta } } } ^ { [ j ] } = \lambda _ { j } \mathbf { J } _ { \boldsymbol { \Theta } } ^ { [ j ] } \mathbf { e } ^ { [ j ] }
$$

Mathematically, these formulations are invariant under the permutation of indices i and $j ,$ imbuing the ensemble of learning laws with group-theoretic properties. While exploiting the algebraic power of group analysis remains outside the scope of this study, it represents a promising avenue for future research;

• Error Reconstruction Pathway. The error propagation for any given layer is governed by

$$
\mathbf { e } ^ { [ i ] } = \mathbb { J } _ { y } ^ { [ i ] } \mathbf { e } _ { y } = \prod _ { k = i } ^ { n } \mathbf { J } _ { y } ^ { [ k ] } \mathbf { e } _ { y }\tag{25}
$$

The layered learning dynamics remain structurally coupled through error reconstruction. In particular, the dynamics of an early layer depend sequentially on the downstream layers.

To exploit the isomorphism topology more efectively, the parallel learning nature of the layered system is mapped into the architecture shown in Fig.3b, which is completely equivalent to Fig.3a. The two distinct features of this setup emerge immediately.

• Global Learning. Every constituent layer executes its parameter learning driven directly by the terminal prediction error $\mathbf { e } _ { y } .$ . Because of this long chain of information transformations, the error signals reaching early layers sufer from severe informational degradation and delay. Consequently, optimization for these layers is highly ineficient. Another critical consequence of this vulnerability is that newly integrated intermediate layers cannot adapt dynamically, as they are starved of localized, high-fidelity error gradients;

• Cascade Learning. The global routing topology gives rise to a large-scale multiloop optimization hierarchy. Under this configuration, downstream (late-stage) layers function as autonomous inner loops, while upstream (early-stage) layers operate as outer loops. This structural coupling is asymmetric. The inner loops continuously perturb the error spaces of the outer loops, while learning in the outer loops has almost no reciprocal efect on the inner loops.

Figures 3c and 3d present two local realizations of the same layered topology. In both cases, each layer is corrected through a locally reconstructed error signal rather than relying exclusively on the terminal prediction error.

## Large-scale Multiple Loop Dynamics

Recursive Error State Dynamics. Based on the global and cascaded learning structure, let the forward mapping of each layer be defined sequentially as:

$$
\mathbf { y } ^ { [ i ] } = f \mathbb { N } \mathbb { N } _ { i } ( \mathbf { y } ^ { [ i - 1 ] } , \boldsymbol { \Theta } ^ { [ i ] } )
$$

which is essentially the combined version of Eqn(2). The parameter update law is governed by Eqn(24) as:

$$
\begin{array} { r } { \dot { \Theta } ^ { [ i ] } = \lambda _ { i } \mathbf { J } _ { \theta } ^ { [ i ] } \mathbf { e } ^ { [ i ] } } \end{array}
$$

where $\pmb \theta ^ { [ i ] } \in \mathbb { R } ^ { n _ { i } \times n _ { i - 1 } }$ . For a layer $i < n ,$ the localized error vector $\mathbf { e } ^ { [ i ] } \in \mathbb { R } ^ { n _ { i } }$ is generated recursively by projecting the intermittent downstream error through the adjacent layer-to-layer state Jacobian matrix

![](images/d35ac2fac108597c87bd04c7a5bc64c007d02c369a0309fd590db8a34aded263.jpg)

![](images/27e4add97ad3b3b37b20785879e46144af8a315eca02eae0116f1fce6c5609fd.jpg)

a) Topologically Isomorphism of Parallel Learning  
![](images/5b6fdf3b885869f6566397635498b0994c1391f3c9e540fe80254dd6386f5e18.jpg)  
c) Topologically Isomorphism of Equivalent Local Learning

b) Topologically Isomorphism of Global Learning  
![](images/1e20178d9739242ce8ef1b07bba8ad499c9123d5812416df10b1fa59bc4e0daf.jpg)  
Fig. 3 Topological Isomorphism of Feedback Learning  
d) Topologically Isomorphism of Predictive Coding Local Learning

$\mathbf { J } _ { y } ^ { [ i ] }$ . This error vector $\mathbf { e } ^ { [ i ] }$ is given in Eqn(25) as:

$$
\mathbf { e } ^ { [ i ] } = \mathbb { J } _ { y } ^ { [ i ] } \mathbf { e } _ { y }
$$

The global tracking mismatch vector $\mathbf { e } _ { y }$ driving the parameter learning is defined by Eqn(5) as:

$$
\mathbf { e } _ { y } = \mathbf { y } _ { s } - \hat { \mathbf { y } } _ { s }
$$

Cascaded Multi-Loop Dynamics. To evaluate parameter convergence, the parameter error defined in Eqn(22) is adopted as

$$
\delta \boldsymbol { \Theta } = \boldsymbol { \Theta } ^ { * } - \boldsymbol { \Theta }
$$

where ${ \mathfrak { \theta } } ^ { * }$ denotes the ideal parameter at equilibrium $\left( { \dot { \mathbf { \theta } } } ^ { * } = \mathbf { 0 } \right)$ , which changes only slowly by nature. Taking the time derivative, yields

$$
\delta { \dot { \Theta } } = - { \dot { \Theta } }
$$

Hence, the dynamic equation for parameter deviation at layer i as

$$
\delta \dot { \mathbf { \theta } } ^ { [ i ] } = - \lambda _ { i } \mathbb { J } _ { \mathbf { \theta } } ^ { [ i ] } \mathbf { e } _ { y }\tag{26}
$$

Expanding this expression layer-by-layer, results in that:

$$
\begin{array} { r l } { \mathrm { L a y e r ~ n ~ : } \qquad \boldsymbol { \delta \dot { \Theta } } ^ { [ n ] } = - \lambda _ { n } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ n ] } \mathbf { e } _ { y } } & { { } = - \lambda _ { n } \mathbf { J } _ { \boldsymbol { \Theta } } ^ { [ n ] } \mathbf { e } _ { y } } \end{array}
$$

$$
\begin{array} { r l r l } { \mathrm { L a y e r ~ j ~ : } } & { { } \quad } & { \delta \dot { \Theta } _ { \hphantom { ( } \hphantom { ( } \cdot \mathrm { ~  ~ \mu ~ } } ^ { [ j ] } = - \lambda _ { j } \mathbb { J } _ { \theta } ^ { [ j ] } \mathbf { e } _ { y } } & { } & { { } = - \lambda _ { j } \mathbf { J } _ { \theta } ^ { [ j ] } \prod _ { k = j } ^ { n } \mathbf { J } _ { y } ^ { [ k ] } \mathbf { e } _ { y } } \end{array}
$$

$$
\begin{array} { r l r l } { \mathrm { L a y e r ~ i ~ : } } & { { } \quad } & { \delta \dot { \Theta } ^ { \lfloor i \rfloor } = - \lambda _ { i } \mathbb { J } _ { \dot { \Theta } } ^ { [ i ] } \mathbf { e } _ { y } } & { } & { { } = - \lambda _ { i } \mathbf { J } _ { \dot { \Theta } } ^ { [ i ] } \prod _ { k = i } ^ { n } \mathbf { J } _ { y } ^ { [ k ] } \mathbf { e } _ { y } } \end{array}
$$

$$
\begin{array} { r l r l } { \mathrm { L a y e r \ 2 : } } & { { } \ } & { \delta \dot { \Theta } _ { \mathrm { . . } } ^ { [ 2 ] } = - \lambda _ { 2 } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ 2 ] } \mathbf { e } _ { y } } & { } & { { } = - \lambda _ { 2 } \mathbf { J } _ { \boldsymbol { \Theta } } ^ { [ 2 ] } \prod _ { k = 2 } ^ { n } \mathbf { J } _ { y } ^ { [ k ] } \mathbf { e } _ { y } } \end{array}
$$

$$
\begin{array} { r l r l } { \mathrm { L a y e r \ 1 \ : } } & { { } \ } & { \delta \dot { \boldsymbol { \Theta } } ^ { [ 1 ] } = - \lambda _ { 1 } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ 1 ] } \mathbf { e } _ { y } } & { } & { { } = - \lambda _ { 1 } \mathbf { J } _ { \boldsymbol { \Theta } } ^ { [ 1 ] } \prod _ { k = 1 } ^ { n } \mathbf { J } _ { y } ^ { [ k ] } \mathbf { e } _ { y } } \end{array}
$$

According to algebraic Eqn(21), and introducing a diagonal error tracking factor $\zeta _ { y } ,$ the dynamic version then becomes

$$
\dot { \mathbf { e } } _ { y } = \mathbf { J _ { \boldsymbol { \Theta } } ^ { [ 1 ] ^ { T } } } \boldsymbol { \delta } \dot { \mathbf { e } } ^ { [ 1 ] } + \mathbf { J _ { \boldsymbol { \Theta } } ^ { [ 2 ] ^ { T } } } \boldsymbol { \delta } \dot { \mathbf { e } } ^ { [ 2 ] } + \cdots + \mathbf { J _ { \boldsymbol { \Theta } } ^ { [ i ] ^ { T } } } \boldsymbol { \delta } \dot { \mathbf { e } } ^ { [ i ] } + \cdots + \mathbf { J _ { \boldsymbol { \Theta } } ^ { [ j ] ^ { T } } } \boldsymbol { \delta } \dot { \mathbf { e } } ^ { [ j ] } + \cdots + \mathbf { J _ { \boldsymbol { \Theta } } ^ { [ n ] ^ { T } } } \boldsymbol { \delta } \dot { \mathbf { e } } ^ { [ j ] } - \mathbb { K } _ { y } \mathbf { e } _ { y }\tag{27}
$$

Stack the above vector diferential equations, from layer 1 and layer 2, through layer i and layer $j ,$ up to layer $n ,$ and combine the error dynamics described by Eqn(27), into a clean matrix format of

$$
\begin{array} { r } { \hat { \delta } \hat { \Phi } ^ { [ i ] } } \\ { \vdots } \\ { \hat { \delta } \hat { \Phi } ^ { [ i ] } } \\ { \vdots } \\ { \hat { \delta } \hat { \Phi } ^ { [ i ] } } \\ { \vdots } \\ { \hat { \delta } \hat { \Phi } ^ { [ i ] } } \\ { \vdots } \\ { \hat { \delta } \hat { \Phi } ^ { [ i ] } } \\ { \vdots } \\ { \hat { \delta } \hat { \Phi } ^ { [ i ] } } \\ { \hat { \delta } ^ { [ i ] } } \\ { \hat { \delta } _ { \small { \small { \bf { p } } } } ^ { [ i ] } } \end{array} \right] = \left[ \begin{array} { c c c c c c c c c c c c c c c c c c c } { \textstyle { \bf { 0 } } } & { \textstyle \mathrm { ~ 0 } } & { \cdots } & { \textstyle \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { - } \lambda _ { 1 } \mathrm { \bf { J } } _ { 0 } ^ { [ ] } \prod _ { k = } ^ { 2 } \mathrm { \bf { J } } _ { y } ^ { [ k ] } } \\ { \mathrm { 0 } } & { \small \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ \small { J } } _ { 0 } ^ { [ ] } \prod _ { k = } ^ { 2 } \mathrm { \bf { J } } _ { y } ^ { [ k ] } } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { \mathrm { ~ 0 } } & { \small \mathrm { ~ 0 } } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } & { \mathrm { ~ 0 } } & { \cdots } &  \mathrm  \end{array}\tag{28}
$$

Note that A is a block matrix, since the parameter state $\delta \Theta ^ { [ i ] }$ has size $n _ { i } \times n _ { i - 1 }$ and the error state $\mathbf { e } _ { y }$ has size $n _ { n } \times 1 . \mathrm { ~ A ~ }$ compact version of A is

$$
\mathbf { A } =  [ \begin{array} { l l l l l l l l l l } { \mathbf { 0 } } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { - \lambda _ { 1 } \mathbf { J } _ { 0 } ^ { [ 1 ] } } \\ { \mathbf { 0 } } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { - \lambda _ { 2 } \mathbf { J } _ { 0 } ^ { [ 2 ] } } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { \mathbf { 0 } } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { - \lambda _ { i } \mathbf { J } _ { 0 } ^ { [ 1 ] } } \\ { \vdots } & { \cdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { \mathbf { 0 } } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { - \lambda _ { j } \mathbf { J } _ { 0 } ^ { [ j ] } } \\ { \vdots } & { \cdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { \vdots } & { \cdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } \\ { \mathbf { 0 } } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { \cdots } & { \mathbf { 0 } } & { - \lambda _ { m } \mathbf { J } _ { 0 } ^ { [ 1 ] } } \\ { \vdots } & { \cdots } & { \vdots } & { \vdots } & { \vdots } &  \vdots  \end{array}
$$

This matrix representation reveals several characteristics of the system, providing additional insight from a dynamic point of view.

• P1: Zero Direct Causal Coupling. The entire upper-left sub-matrix consists exclusively of zero blocks, demonstrating the absence of direct causal coupling between the layered parameter spaces. This decoupling is formally established by the following cross-sensitivity functions.

$$
\frac { \partial \delta \dot { \Theta } ^ { [ i ] } } { \partial \delta \Theta ^ { [ j ] } } = \mathbf { 0 } \quad ( \forall i \neq j ) , \qquad \frac { \partial \delta \dot { \Theta } ^ { [ i ] } } { \partial \mathbf { e } _ { y } } = - \lambda _ { i } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] }
$$

Clearly, the derivative of $\delta \Theta ^ { [ i ] }$ does not depend on the parameter state of $\delta \Theta ^ { [ j ] }$ of any other layer. This structural independence guarantees zero direct causal coupling between distinct layers. However, the layers remain implicitly interconnected through the tracking error feedback network. The parameter deviations δθ of individual layers drive the reduction of the tracking error $\mathbf { e } _ { y }$ collectively. Simultaneously, the emergent error $\mathbf { e } _ { y }$ propagates back to govern the evolution of $\delta \Theta$ across all layers. A perturbation in $\delta \Theta ^ { [ i ] }$ can therefore influence $\delta \Theta ^ { [ j ] }$ only through the shared error state $\mathbf { e } _ { y } .$ This architecture efectively insulates the system against cross-destabilization within the internal parameter subspaces during adaptation.

• P2: Skew-Symmetric Cross-Coupling. The cross-coupling blocks of A are skew-symmetric up to the learning-rate scaling. Let

$$
\mathbb { J } _ { s k } = [ \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ 1 ] } , \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ 2 ] } , \cdot \cdot \cdot , \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } , \cdot \cdot \cdot , \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ n ] } ] ^ { T }
$$

and

$$
\boldsymbol { \Lambda } = \operatorname { d i a g } ( \lambda _ { 1 } I , \lambda _ { 2 } I , \ldots , \lambda _ { n } I )
$$

Then A can be written in a more compact block form.

$$
\mathbf { A } = \left[ \begin{array} { l l } { \mathbf { 0 } } & { - \Lambda \mathbf { J } _ { s k } } \\ { \mathbf { J } _ { s k } ^ { T } } & { - \mathbb { K } _ { y } } \end{array} \right]\tag{29}
$$

Scaled by the learning rates λ, the cross-terms satisfy $A _ { 1 2 } = - \Lambda A _ { 2 1 } ^ { T }$ . This skew-symmetric type of matrix represents a passive system in general. Let a system given by $\dot { \Theta } = \mathbb { A } _ { s k } \Theta$ , with

$$
\mathbb { A } _ { s k } = \left[ \mathbf { 0 } _ { \Delta t } - \Delta \mathbf { J } _ { s k } \right] , \quad \quad \mathbf { W } = \left[ \begin{array} { c c } { \mathbf { A } ^ { - 1 } \mathbf { 0 } } \\ { \mathbf { 0 } } & { \mathbf { I } } \end{array} \right] , \quad \quad \check { \mathbb { A } } _ { s k } = \mathbf { W } \mathbb { A } _ { s k } = \left[ \begin{array} { c c } { \mathbf { 0 } } & { - \mathbf { J } _ { s k } } \\ { \mathbf { J } _ { s k } ^ { T } } & { \mathbf { 0 } } \end{array} \right]
$$

and introduce an energy function $\mathcal { L } _ { \Theta }$ as $\begin{array} { r } { \mathcal { L } _ { \Theta } = \frac { 1 } { 2 } \Theta ^ { T } \mathbf { W } \Theta } \end{array}$ , Then

$$
\begin{array} { r l r l } { \dot { \mathcal { L } } _ { \Theta } = } & { } & { \Theta ^ { T } \mathbf { W } \dot { \Theta } } & & { ( \mathbf { W } \mathrm { ~ i s ~ p o s i t i v e ~ d e f i n i t e } ) } \\ { = } & { } & { \Theta ^ { T } \mathbf { W } \mathbb { A } _ { s k } \Theta } \\ { = } & { } & { \Theta ^ { T } \check { \mathbb { A } } _ { s k } \Theta } & { \mathrm { T a k e ~ a ~ t r a n s p o s e } } \\ { = - } & { \Theta ^ { T } \check { \mathbb { A } } _ { s k } \Theta } & & { \mathrm { S t r i c t l y ~ s k e w - s y m m e t r i c } } \\ { = } & { 0 } & & { x = - x \mapsto x = 0 } \end{array}\tag{30}
$$

This implies that $\mathcal { L } _ { \Theta }$ is constant. Thus, a skew-symmetric system is energy conservative and marginally stable. This is the case for parameter learning, and the associated dynamics of each layer are marginally stable.

• P3: Strictly Damped Tracking Error Mode. The last block $\mathbb { K } _ { y }$ at bottom-right ensures that the global tracking error $\mathbf { e } _ { y }$ experiences strict negative feedback, thereby facilitating system energy dissipation. With the energy function been defined as Eqn(8), substituting Eqn(26) and Eqn(27), its derivative then becomes

$$
\begin{array} { r l } & { \dot { \mathcal { L } } _ { e } = \mathrm { e } _ { y } ^ { T } \dot { \mathbf { e } } _ { y } } \\ &  \quad = \mathrm { e } _ { y } ^ { T } \left( { \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ 1 ] ^ { T } } { \delta \dot { \boldsymbol { \Theta } } } ^ { [ 1 ] } + \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ 2 ] ^ { T } } { \delta \dot { \boldsymbol { \Theta } } } ^ { [ 2 ] } + \cdots + \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] ^ { T } } { \delta \dot { \boldsymbol { \Theta } } } ^ { [ i ] } + \cdots + \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ n ] ^ { T } } { \delta \dot { \boldsymbol { \Theta } } } ^ { [ n ] } - \mathbb { K } _ { y } \mathbf { e } _ { y } \right) } \\ & { \quad = \mathrm { e } _ { y } ^ { T } \left( - \lambda _ { 1 } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ 1 ] ^ { T } } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ 1 ] } - \lambda _ { 2 } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ 2 ] ^ { T } } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ 2 ] } - \cdots - \lambda _ { i } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ i ] ^ { T } } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ i ] } - \cdots - \lambda _ { n } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ n ] ^ { T } } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ n ] } - \mathbb { K } _ { y } \right) \mathbf { e } _ { y } } \\ & { \quad \le 0 } \end{array}\tag{31}
$$

Introducing the damping gain $\mathbb { K } _ { y }$ further enhances energy dissipation within the system. To illustrate this efect, the system matrix A is decomposed into a skew-symmetric component $\mathbb { A } _ { s k }$ and a negative semi-definite damping block $\mathbf { \Lambda } _ { \mathbf { \Lambda } _ { D } } = \mathrm { d i a g } ( \mathbf { 0 } , - \mathbb { K } _ { y } )$ , where ${ \mathbb K } _ { y } \succ 0$ . Define $\check { \mathbb { D } } _ { s y } = \mathbf { W } \mathbf { A } _ { D }$ . This decomposition yields

$$
\begin{array} { r l } { \dot { \mathcal { L } } _ { \Theta } = } & { \Theta ^ { T } \mathbf { W } \dot { \Theta } } \\ { = } & { \Theta ^ { T } \mathbf { W } \mathbb { A } _ { s k } \Theta + \Theta ^ { T } \mathbf { W } \boldsymbol { \Lambda } _ { D } \Theta } \\ { = } & { \underbrace { \Theta ^ { T } \check { \mathbb { A } } _ { s k } \Theta } _ { \mathrm { C o n s e r v a t i v e ~ ( = 0 ) } } + \underbrace { \Theta ^ { T } \check { \mathbb { D } } _ { s y } \Theta } _ { \mathrm { D i s s i p a t i v e ~ ( \le 0 ) } } } \\ { \le } & { 0 } \end{array}
$$

Consequently, although the skew-symmetric term $\mathbb { J } _ { s k }$ reduces the energy associated with the tracking error, it merely redistributes into the parameter space to maintain a system level of energy balance. Ultimately, it is the injection of the damping matrix $\mathbb { K } _ { y }$ that actively dissipates net energy from the system.

Properties P1 and P2 demonstrate that each learning loop outlined in Fig. 3b operates globally through the terminal layer. Specifically, inter-layer learning coupling occurs exclusively via the global tracking error, yielding marginal loop stability dictated by energy conservation. Conversely, Property P3 establishes that each individual loop achieves asymptotic stability when isolated from the cascaded network and integrated with error tracking. From a dynamical systems perspective, this Online Active Learning (OAL) architecture can be interpreted as having a single stable outer loop (conventionally chosen as Layer 1), while all remaining loops operate as inner, marginally stable loops. This interpretation reveals a fundamental structural isomorphism: first, any arbitrary loop can topologically serve as the stable outer loop; second, all internal loops are strictly isomorphic with respect to both their learning topology and their marginal stability characteristics. Structurally, the outer loop dynamics govern the system’s forced response, driving the tracking error asymptotically to zero under the control of the damping factor $\mathbb { K } _ { y }$ . In contrast, the internal loops define the unforced zero-dynamics [21] of the system, dictating the autonomous parameter trajectories that drift and freeze along a continuous equilibrium manifold once the tracking error vanishes.

Spectral Analysis and Parameter Subspace Properties. To evaluate the underlying learning dynamics, the eigenspace of matrix A is analyzed adopting the compact-block representation as Eqn(29). Solving the characteristic equation $\mathbf { A } \mathbf { V } = \mu \mathbf { V }$ for the partitioned eigenvector $\mathbf { \bar { V } } = [ \mathbf { V } _ { \theta } ^ { T } , \mathbf { V } _ { y } ^ { T } ] ^ { T }$ yields two

coupled algebraic constraints as

$$
- \Lambda \mathbb { J } _ { s k } \mathbf { V } _ { y } = \mu \mathbf { V } _ { \boldsymbol { \theta } }\tag{32}
$$

$$
\begin{array} { r } { { \bf J } _ { s k } ^ { T } { \bf V } _ { \theta } - \mathbb { K } _ { y } { \bf V } _ { y } = \mu { \bf V } _ { y } } \end{array}\tag{33}
$$

Eqn(32) and Eqn(33) reveals that the spectrum maps directly to the two functional sub-spaces as: 1) the stabilized space of the outer loop dynamic; and 2) the null-space of the inner zero dynamics.

• Stabilized Outer Loop Dynamics. For the non-zero spectrum $( \mu \neq 0 )$ governing the transient tracking error, substituting $\begin{array} { r } { \dot { \mathbf { V } } _ { \boldsymbol { \Theta } } = - \frac { 1 } { \mu } \pmb { \Lambda } \mathbf { J } _ { s k } \mathbf { V } _ { y } } \end{array}$ from Eqn(32) directly into Eqn(33) isolates the outer loop as

$$
\left( \mathbb { J } _ { s k } ^ { T } \pmb { \Lambda } \mathbb { J } _ { s k } \right) \mathbf { V } _ { y } = - \mu ( \mu \mathbf { I } + \mathbb { K } _ { y } ) \mathbf { V } _ { y }\tag{34}
$$

Let $\sigma ^ { 2 } > 0$ denote an eigenvalue of the symmetric and positive-definite matrix $\mathbf { M } _ { c } = \mathbf { J } _ { s k } ^ { T } \mathbf { A } \mathbf { J } _ { s k }$ which represents the aggregate, learning-rate-scaled sensitivity of the cascaded layers. Assuming a scalar configuration for the damping factor $\mathbb { K } _ { y } = k _ { y } \mathbf { I }$ , this relationship collapses to a scalar quadratic characteristic equation:

$$
\mu ^ { 2 } + k _ { y } \mu + \sigma ^ { 2 } = 0
$$

This proves that the forced response of the outer tracking loop is stabilized, since the real parts of these eigenvalues are strictly negative, $\mathrm { R e } ( \mu ) < 0$ , as a result of the designed damping factor $k _ { y } ~ > ~ 0$ . Furthermore, the system can be explicitly designed to exhibit over-damped $( k _ { y } ^ { 2 } \ge 4 \sigma ^ { 2 } )$ characteristics if required.

• Zero Dynamics of the Inner Null-Space. From a nonlinear dynamics perspective, evaluating the null-spectrum $( \mu = 0 )$ characterizes the unforced zero-dynamics [21] of the OAL system, specifically representing the internal parameter adaptations when the tracking error is held to zero. Setting $\mu = 0$ in Eqn(32) directly dictates that the tracking error component vanishes $( \mathbf { V } _ { y } = \mathbf { 0 } )$ . Substituting this into Eqn(33) formalizes the zero-dynamics eigenspace as

$$
{ \bf V } _ { 0 } = [ { \bf V } _ { \oplus } , { \bf 0 } ] ^ { T } \quad \mathrm { s u b j e c t ~ t o } \quad \mathbb { J } _ { s k } ^ { T } { \bf V } _ { \oplus } = { \bf 0 }\tag{35}
$$

Consequently, the parametric zero dynamics locally reside within the null space of $\mathbb { J } _ { s k }$ . Because the collective parameter space is overwhelmingly large relative to the tracking output space, $\mathbb { J } _ { s k }$ fundamentally possesses a high-dimensional kernel, namely ker $\left( \mathbb { J } _ { s k } \right)$ . Without an additional mechanism to break this structural symmetry, internal parameter updates stall along the manifold explicitly characterized by Eqn(23). Thus, when the outer loop collapses to zero tracking error, the internal zero dynamics are only marginally stable. This permanently locks the learnable parameters into an infinite number of sub-optimal combinations within the just-mentioned null space.

## Internal Zero Dynamics Stabilization

To guarantee global parameter convergence where $\delta \Theta ^ { [ i ] } ( t ) \to \mathbf { 0 }$ as $t \to \infty$ across all layers $i ,$ the system must disrupt the stationary high-dimensional manifold defined by Eqn(23) that resides within the kernel of $\mathbb { A } _ { s k }$ . This intervention fundamentally alters the system properties, forcing the internal zero-dynamics to shift toward asymptotic stability via the Persistency of Excitation (PE) condition.

Persistency of Excitation Method. Within the $\mathbf { O A L }$ framework, the dynamic streaming of input signals forces the composite cascaded Jacobian matrix $\mathbb { J } _ { s k } ( t )$ to evolve as a time-dependent component. Specifically, if $\mathbb { J } _ { s k } ( t )$ satisfies the classical PE constraint:

$$
\int _ { t } ^ { t + T } \mathbb { J } _ { s k } ^ { T } ( \tau ) \mathbb { J } _ { s k } ( \tau ) d \tau \geq \alpha \mathbf { I } > 0\tag{36}
$$

for $T > 0$ and $\alpha > 0$ . This condition efectively rotates the high-dimensional kernel across the parameter space, preventing the state trajectories from settling into sub-optimal equilibria. Thus, parametric adaptation remains active until the error vector is explicitly driven to $\delta \mathbf { \boldsymbol { \Theta } } = \mathbf { \boldsymbol { 0 } }$ . Importantly, this requirement is stronger than pointwise non-vanishing Jacobians. The local sensitivities must vary over the finite hori zon so that their accumulated Gramian is full rank; otherwise, the learning trajectory may still freeze in a fixed null direction.

Dynamically, satisfying the PE condition removes the residual zero-dynamics over a finite time window. By strengthening the energy argument from pointwise non-increase, $\dot { \mathcal { L } } \leq 0 .$ , to finite-window contraction, this mechanism dissipates the residual energy associated with the null space. Instead of allowing parameters to drift along a flat, marginally stable valley, suficient excitation prevents the output error from remaining zero unless the parameter errors also vanish. This upgrades the internal learning dynamics from marginal stability to asymptotic parameter convergence under the PE condition.

Active Damping Approach. Although the PE condition can remove the residual zero dynamics, it remains a passive requirement that depends entirely on external excitation. When input signals lack suficient variation, parameter adaptation can still become trapped within the null space. To counteract these situations, active damping for parameter learning, as introduced in Eqn(17), Eqn(18), and Eqn(19), is injected to dissipate the energy associated with internal parametric fluctuations. This intervention introduces a dedicated leakage matrix, $\mathbb { K } _ { \theta } ^ { [ i ] } > 0$ , directly into the diagonal blocks of the state-space matrix A from Eqn(28). Under this configuration, the parameters follow the augmented dynamic update law:

$$
\begin{array} { r } { \dot { \mathbf { \theta } } ^ { [ i ] } = - \mathbb { K } _ { \mathbf { \theta } } ^ { [ i ] } \mathbf { \theta } ^ { [ i ] } + \lambda _ { i } \mathbb { J } _ { \mathbf { \theta } } ^ { [ i ] } \mathbf { e } _ { y } } \end{array}\tag{37}
$$

which serves as the regularized counterpart to the nominal dynamics in Eqn(24). From an algorithmic perspective, the discrete-time implementation of this iterative structure closely mirrors weight decay in standard optimization. This mechanism alters the residual zero dynamics of the inner loops by introducing strict energy dissipation into the kernel of the composite Jacobian $\mathbb { J } _ { s k }$ . Consequently, even when the tracking error vanishes $( { \bf e } _ { y } = { \bf 0 } )$ , the autonomous parameter trajectories are pulled toward a bounded region, preventing parametric drift. From an ofline-training perspective, this damping acts as weight decay or $\scriptstyle L _ { 2 } \to \operatorname { s t y l e }$ regularization, rather than parameter normalization. Whereas momentum accelerates ofline convergence [22], online damping stabilizes the adaptive loop against excitation deficits.

## Cascade Convergence and Stability Analysis

State-Space Representation. Incorporating the active damping matrices for parameter learning as diagonal elements across the sequential layer blocks yields the following updated version of Eqn(28) as

$$
\begin{array} { r } { [ \delta \dot { { \bf { \theta } } } ^ { [ 1 ] } ] } \\ { \dot { \delta { \bf { \theta } } } ^ { [ 2 ] } } \\ { \dot { \langle \bf { \theta } \rangle } } \\ { \dot { \delta { \bf { \theta } } } ^ { [ i ] } } \\ { \dot { \langle \bf { \theta } \rangle } } \\ { \dot { \delta { \bf { \theta } } } ^ { [ n ] } } \\ { \dot { \delta { \bf { \theta } } } ^ { [ n ] } } \\ { \dot { \mathrm { { \scriptsize ~ c } } } ^ { [ n ] } } \end{array} ] = [ \begin{array} { c c c c c c } { - \mathbb { K } _ { 0 } ^ { [ 1 ] } } & { { \bf { 0 } } } & { \cdots } & { { \bf { 0 } } } & { \cdots } & { { \bf { 0 } } } & { - \lambda _ { 1 } { \bf { J } } _ { \bf { \theta } } ^ { [ 1 ] } } \\ { { \bf { 0 } } } & { - \mathbb { K } _ { \boldsymbol { \theta } } ^ { [ 2 ] } } & { \cdots } & { { \bf { 0 } } } & { \cdots } & { { \bf { 0 } } } & { - \lambda _ { 2 } { \bf { J } } _ { \bf { \theta } } ^ { [ 2 ] } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } & { \ddots } & { \vdots } & { \vdots } \\ { { \bf { 0 } } } & { { \bf { 0 } } } & { \ldots } & { - \mathbb { K } _ { \boldsymbol { \theta } } ^ { [ i ] } } & { \ldots } & { { \bf { 0 } } } & { - \lambda _ { i } { \bf { J } } _ { \bf { \theta } } ^ { [ i ] } } \\ { { \bf { 0 } } } & { \vdots } & { \ddots } & { \vdots } & { \ddots } & { \vdots } & { \vdots } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } & { \ddots } & { \vdots } & { \vdots } \\ { { \bf { 0 } } } & { { \bf { 0 } } } & { \ldots } & { { \bf { 0 } } } & \end{array}\tag{38}
$$

where $\mathbb { K } _ { \theta } ^ { [ i ] } > 0$ denotes the active damping for Layer $i ,$ and $\mathbb { J } _ { \boldsymbol { \theta } } ^ { [ i ] }$ denotes the lumped Jacobian matching the cascaded structural topology. To establish global asymptotic stability across this multi-loop architecture, the topological isomorphism method is adopted. Because the parameter learning topology is identical across all layers, evaluating the stability of a generalized Layer i from a dynamical systems perspective functionally validates the stability properties of the entire unified system.

Parameter Learning Local Stability. To establish the learning stability, layer i is isolated locally. All upstream and downstream parameter errors $( \delta \boldsymbol { \Theta } ^ { [ 1 ] } , \dots , \delta \boldsymbol { \Theta } ^ { [ k ] } , \dots , \delta \boldsymbol { \Theta } ^ { [ n ] }$ , for $k \neq i )$ are treated as bounded external disturbances. A local Lyapunov-like energy function $\mathcal { L } _ { i } ( t )$ is defined as

$$
\mathcal { L } _ { i } ( t ) = \frac { 1 } { 2 } \mathbf { e } _ { y } ^ { T } \mathbf { e } _ { y } + \frac { 1 } { 2 \lambda _ { i } } \delta \mathbf { \Theta } ^ { [ i ] ^ { T } } \delta \mathbf { \Theta } ^ { [ i ] }\tag{39}
$$

The derivative of $\mathcal { L } _ { i } ( t )$ along the state trajectories is

$$
\dot { \mathcal { L } } _ { i } ( t ) = { \bf e } _ { y } ^ { T } \dot { \bf e } _ { y } + \frac { 1 } { \lambda _ { i } } \delta { \bf \Theta } ^ { [ i ] ^ { T } } \delta \dot { \bf \Theta } ^ { [ i ] }
$$

Substituting the corresponding row dynamics from (38) for $\dot { \mathbf { e } } _ { y }$ and $\delta \dot { \Theta } ^ { [ i ] }$ yields:

$$
\begin{array} { l } { { \displaystyle \dot { \mathcal { L } } _ { i } ( t ) = - \mathbf { e } _ { y } ^ { T } \mathbb { K } _ { y } \mathbf { e } _ { y } + \mathbf { e } _ { y } ^ { T } \left( \sum _ { k = 1 , k \neq i } ^ { n } \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ k ] ^ { T } } \delta \boldsymbol { \Theta } ^ { [ k ] } + \mathbb { I } _ { \boldsymbol { \Theta } } ^ { [ i ] ^ { T } } \delta \boldsymbol { \Theta } ^ { [ i ] } \right) - \frac { 1 } { \lambda _ { i } } \delta \boldsymbol { \Theta } ^ { [ i ] ^ { T } } \left( \lambda _ { i } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \mathbf { e } _ { y } + \mathbb { K } _ { \boldsymbol { \Theta } } ^ { [ i ] } \delta \boldsymbol { \Theta } ^ { [ i ] } \right) } } \\ { { \displaystyle \qquad = - \mathbf { e } _ { y } ^ { T } \mathbb { K } _ { y } \mathbf { e } _ { y } - \frac { 1 } { \lambda _ { i } } \delta \mathbf { e } ^ { [ i ] ^ { T } } \mathbb { K } _ { \boldsymbol { \Theta } } ^ { [ i ] } \delta \boldsymbol { \Theta } ^ { [ i ] } + \breve { \Delta } _ { i } ( t ) } } \end{array}
$$

where the cross-layer coupling term $\check { \Delta } _ { i } ( t )$ is defined as:

$$
\breve { \pmb { \Delta } } _ { i } ( t ) = \mathbf { e } _ { y } ^ { T } ( t ) \sum _ { k = 1 , k \neq i } ^ { n } \mathbb { J } _ { \pmb { \theta } } ^ { [ k ] ^ { T } } \delta \pmb { \theta } ^ { [ k ] }
$$

This term represents the multi-layer disturbance generated by the residual learning errors of the remaining cascaded layers. For an isolated layer, $\check { \Delta } _ { i } ( t )$ must therefore be treated as an external coupling term rather than eliminated locally. The local negative-definite part is

$$
\dot { \mathcal { L } } _ { i } ( t ) - \check { \mathbf { A } } _ { i } ( t ) = - \left( \mathbf { e } _ { y } ^ { T } \mathbb { K } _ { y } \mathbf { e } _ { y } + \frac { 1 } { \lambda _ { i } } \delta \mathbf { \Theta } ^ { [ i ] ^ { T } } \mathbb { K } _ { \mathbf { \Theta } _ { 0 } } ^ { [ i ] } \delta \mathbf { \Theta } \mathbf { \Theta } ^ { [ i ] } \right) \le 0\tag{40}
$$

If the residual coupling is absent, or after the complete cascaded cancellation shown in the next subsection is applied, the derivative is non-positive and vanishes only when both the tracking error and the local parameter error vanish $( { \bf e } _ { y } \ = \ { \bf 0 }$ and $\delta \boldsymbol { \Theta } ^ { [ i ] } \ : = \ : \mathbf { 0 } )$ . Thus, this local calculation identifies the damped layer-wise stability mechanism, while the full asymptotic conclusion requires the global cascaded proof below.

From a physical perspective, $\mathbb { K } _ { y }$ acts as a virtual damping that dissipates energy within the tracking loop (which represents a topological rather than a physical loop), while $\mathbb { K } _ { \boldsymbol { \theta } } ^ { [ i ] }$ mitigates the energy associated with the parameter adaptation loop of layer i. Notably, the presence of the adaptation gain $\mathbb { K } _ { \boldsymbol { \theta } } ^ { [ i ] }$ renders the tracking gain $\mathbb { K } _ { y }$ non-essential, thereby providing enhanced control design flexibility.

Parameter Learning Cascaded Stability. Because the parameter learning of all layers is coupled through the global error tracking loop, a comprehensive assessment of the integrated system is necessary. This evaluation must simultaneously account for tracking error stability, parameter convergence, and cross-layer learning interactions. To this end, a global Lyapunov candidate function $\mathcal { L } ( t )$ is defined, which is intrinsically a multi-layer extension of (8),

$$
\mathcal { L } ( \mathbf { y } , \boldsymbol { \Theta } , t ) = \frac { 1 } { 2 } \mathbf { e } _ { y } ^ { T } \mathbf { e } _ { y } + \sum _ { i = 1 } ^ { n } \frac { 1 } { 2 \lambda _ { i } } \delta \boldsymbol { \Theta } ^ { [ i ] ^ { T } } \delta \boldsymbol { \Theta } ^ { [ i ] }\tag{41}
$$

Diferentiating $\mathcal { L } ( t )$ along the state trajectories of the full cascaded space yields:

$$
\dot { \mathcal { L } } ( t ) = { \mathbf e } _ { y } ^ { T } \dot { \mathbf e } _ { y } + \sum _ { i = 1 } ^ { n } \frac { 1 } { \lambda _ { i } } \delta { \mathbf { \theta } ^ { [ i ] } } ^ { T } \delta \dot { \mathbf { \theta } } ^ { [ i ] }
$$

Substituting the full system dynamics from (38) expands the time derivative into

$$
\begin{array} { l } { { \displaystyle { \dot { \mathcal { L } } } ( t ) = - { \bf e } _ { y } ^ { T } \mathbb { K } _ { y } { \bf e } _ { y } + { \bf e } _ { y } ^ { T } \sum _ { i = 1 } ^ { n } { \mathbb { J } _ { \Theta } ^ { [ i ] ^ { T } } \delta \Theta ^ { [ i ] } } - \sum _ { i = 1 } ^ { n } \frac { 1 } { \lambda _ { i } } \delta \Theta ^ { [ i ] ^ { T } } \left( \lambda _ { i } { \mathbb { J } _ { \Theta } ^ { [ i ] } } { \bf e } _ { y } + { \mathbb { K } _ { \Theta } ^ { [ i ] } } \delta \Theta ^ { [ i ] } \right) } } \\ { ~ } \\ { { \displaystyle ~ = - { \bf e } _ { y } ^ { T } \mathbb { K } _ { y } { \bf e } _ { y } + \sum _ { i = 1 } ^ { n } { \bf e } _ { y } ^ { T } { \mathbb { J } _ { \Theta } ^ { [ i ] ^ { T } } \delta \Theta ^ { [ i ] } } - \sum _ { i = 1 } ^ { n } \delta \Theta ^ { [ i ] ^ { T } } { \mathbb { J } _ { \Theta } ^ { [ i ] } } { \bf e } _ { y } - \sum _ { i = 1 } ^ { n } \frac { 1 } { \lambda _ { i } } \delta \Theta ^ { [ i ] ^ { T } } \mathbb { K } _ { \Theta } ^ { [ i ] } \delta \Theta ^ { [ i ] } } }  \end{array}
$$

Noting that the cross-layer interaction terms are scalars, they satisfy the transpose identity

$$
\left( \sum _ { i = 1 } ^ { n } \mathbf { e } _ { y } ^ { T } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } { } ^ { T } \delta \boldsymbol { \Theta } ^ { [ i ] } \right) ^ { T } = \sum _ { i = 1 } ^ { n } \delta \boldsymbol { \Theta } ^ { [ i ] ^ { T } } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \mathbf { e } _ { y }
$$

Consequently, these skew-symmetric coupling terms cancel out identically. This cancellation implies that the learning adaptation law perfectly matches and neutralizes the dynamic interactions across the cascaded layers, reducing the derivative to

$$
\dot { \mathcal { L } } ( t ) = - \mathbf { e } _ { y } ^ { T } \mathbb { K } _ { y } \mathbf { e } _ { y } - \sum _ { i = 1 } ^ { n } \frac { 1 } { \lambda _ { i } } \delta \mathbf { \theta } ^ { [ i ] ^ { T } } \mathbb { K } _ { \mathbf { \theta } } ^ { [ i ] } \delta \mathbf { \theta } ^ { [ i ] } \le 0\tag{42}
$$

Eqn(42) establishes that $\mathcal { L } ( t )$ is monotonically non-increasing and bounded from below $( \mathcal { L } ( t ) \geq 0 )$ By invoking Barbalat’s lemma [23] alongside Lyapunov stability criteria [24, 25], it follows that $\begin{array} { r } { \operatorname* { l i m } _ { t  \infty } \dot { \mathcal { L } } ( t ) = 0 } \end{array}$ , implying that the system trajectories eventually converge to a minimal energy state

[26]. Because the coeficient matrices $\mathbb { K } _ { y }$ and $\mathbb { K } _ { \theta } ^ { [ i ] } , i \in \{ 1 , 2 , \dots , n \}$ , of the two remaining quadratic forms in Eqn(42) are strictly positive definite, it follows that

$$
\operatorname* { l i m } _ { t \to \infty } \mathbf { e } _ { y } ( t ) = \mathbf { 0 } \quad { \mathrm { a n d } } \quad \operatorname* { l i m } _ { t \to \infty } \delta \mathbf { \Theta } ^ { [ i ] } ( t ) = \mathbf { 0 } \quad \forall i \in \{ 1 , 2 , \dots , n \}
$$

This completes the proof of asymptotic stability for the cascaded OAL learning dynamics. In particular, Eqn(42) again shows that the tracking gain $\mathbb { K } _ { y }$ is non-essential, justifying its frequent omission in practical implementations.

However, the active damping term $\mathbb { K } _ { \boldsymbol { \theta } } ^ { [ i ] }$ also governs how fast the parameter energy decays. To quantify this efect without loss of generality, consider a simplified scenario in which the tracking gain $\mathbb { K } _ { y }$ is omitted and the damping matrix $\mathbb { K } _ { \boldsymbol { \Theta } } ^ { [ i ] }$ reduces to a uniform scalar factor $\kappa _ { i } = \kappa$ . The energy function in Eqn(41) then reduces to

$$
\mathcal { L } ( t ) = \sum _ { i = 1 } ^ { n } \frac { 1 } { 2 \lambda _ { i } } \delta \mathbf { \Theta } ^ { [ i ] ^ { T } } \delta \mathbf { \Theta } \mathbf { \Theta } ^ { [ i ] }
$$

and its derivative, which is a simplified version of $\mathrm { E q n } ( 4 2 )$ , becomes

$$
\begin{array} { l } { \displaystyle \dot { \mathcal { L } } ( t ) = - \sum _ { i = 1 } ^ { n } \frac { \kappa _ { i } } { \lambda _ { i } } \delta \Theta ^ { [ i ] ^ { T } } \delta \Theta ^ { [ i ] } } \\ { \displaystyle = - 2 \kappa \sum _ { i = 1 } ^ { n } \frac { 1 } { 2 \lambda _ { i } } \| \delta \Theta ^ { [ i ] } \| _ { 2 } ^ { 2 } } \\ { \displaystyle = - 2 \kappa \mathcal { L } ( t ) } \end{array}
$$

The temporal profile of this simplified system energy is governed by the exact analytical solution

$$
\mathcal { L } ( t ) = \mathcal { L } ( 0 ) e ^ { - 2 \kappa t }
$$

This expression shows that the damping factor κ sets the exponential decay rate of the parameter energy. Larger values produce faster decay in this simplified model, although practical tuning must balance retention of learned structure against adaptation to new data.

## Overall OAL learning Stability

While the preceding subsection establishes the asymptotic stability of the cascaded learning framework, overall system convergence still depends on two foundational conditions. First, the ideal parameter must be unique to guarantee well-defined convergence. In particular, this uniqueness refers not to a static parameter state ${ \mathfrak { \theta } } ^ { * }$ , but to a time-varying trajectory ${ \boldsymbol { \Theta } } ^ { * } ( t )$ . The trajectory remains unique at each time $t \in \mathbb { R } ^ { + }$ while encoding environmental dynamics over a given interval. Second, within each layer $i \in \{ 1 , 2 , \ldots , n \}$ , the parameter matrix $\boldsymbol { \Theta } ^ { [ i ] }$ regulates multiple concurrent inference channels. Because cascaded inter-layer coupling is not inherently isomorphic to intra-layer coupling, these parallel learning processes must independently satisfy localized stability criteria to ensure global convergence of the multilayer system.

Parameter Learning Global Stability. The state-space Eqn(38) characterizes the system dynamics exclusively via a localized linearization, meaning the associated lumped Jacobian matrix $\mathbb { J } _ { \boldsymbol { \theta } } ^ { [ i ] }$ captures only the local behaviors. To extend the cascaded learning convergence globally, these Jacobians cannot merely serve as local derivatives. Instead, the underlying global mapping

$$
\begin{array} { r l } { \mathcal { F } _ { \boldsymbol { \Theta } } ^ { [ i ] } : } & { { } \mathfrak { M } _ { \mathbf { y } ^ { [ n ] } } \to \mathfrak { M } _ { \boldsymbol { \Theta } ^ { [ i ] } } } \end{array}\tag{43}
$$

must constitute a global monotonic isomorphism. Specifically, because this mapping is a cascaded composition propagating sequentially through the intermediate latent spaces ${ \mathfrak { M } } _ { \mathbf { y } ^ { [ n - 1 ] } } \times \cdot \cdot \cdot \times { \mathfrak { M } } _ { \mathbf { y } ^ { [ i ] } }$ , it must establish a smooth global difeomorphism across the entire joint topological space. This condition requires that the lumped Jacobian $\mathbb { J } _ { \boldsymbol { \theta } } ^ { [ i ] }$ remains non-singular and strictly sign-preserving across the entire product space, defined as

$$
\Omega ^ { [ i ] } : = \mathfrak { M } _ { \mathbf { y } ^ { [ n ] } } \times \mathfrak { M } _ { \mathbf { y } ^ { [ n - 1 ] } } \times \cdot \cdot \cdot \times \mathfrak { M } _ { \mathbf { y } ^ { [ i ] } } \times \mathfrak { M } _ { \mathbf { \theta } ^ { [ i ] } }
$$

The geometric deduction of the global difeomorphism requirement to the uniqueness of the ideal trajectory ${ \boldsymbol { \Theta } } ^ { * } ( t )$ is fundamentally rooted in the topological architecture of the system energy function.

Referring to the primary formulation of Eqn.(8), the energy function $\mathcal { L } _ { e } ( \boldsymbol { \Theta } ^ { [ i ] } )$ to be minimized is explicitly formulated to isolate the dependency on the layer-wise parameter matrix $\dot { { \bf \theta } } ^ { [ i ] } ( t )$

$$
\begin{array} { r l } & { \mathcal { L } _ { e } ( \mathbf { \boldsymbol { \theta } } ^ { [ i ] } ) = \frac { 1 } { 2 } \| \mathbf { y } ( t ) - f \mathbb { N } \mathbb { N } _ { \mathrm { d l } } ^ { [ i ] } [ \mathbf { \boldsymbol { \theta } } ^ { [ i ] } ( t ) ] \| ^ { 2 } . } \\ & { \quad \quad \quad = \frac { 1 } { 2 } \| \mathbf { y } ( t ) - f \mathbb { N } \mathbb { N } _ { n } \circ \cdot \cdot \cdot \circ f \mathbb { N } \mathbb { N } _ { i - 1 } \circ f \mathbb { N } \mathbb { N } _ { i } [ \mathbf { \boldsymbol { \theta } } ^ { [ i ] } ( t ) ] \| ^ { 2 } } \end{array}
$$

This defines the mapping $\mathcal { L } _ { e } : \mathfrak { M } _ { \pmb { \theta } ^ { [ i ] } } \times \mathfrak { M } _ { \mathbf { y } ^ { [ n ] } } \to \mathbb { R } ^ { + }$ , representing the energy landscape under a squared Euclidean distance metric.

Because $\mathcal { L } _ { e } ^ { i }$ prescribes a strictly convex parabolic geometry with respect to the prediction error $\mathbf { e } _ { y } ,$ the underlying topology of the ideal parameter $\boldsymbol { \Theta } _ { * } ^ { [ i ] }$ is dictated entirely by the properties of the neural mapping $f \mathbb { N } _ { \mathrm { d l } } ^ { [ i ] }$ . The first-order necessary condition for optimality requires the gradient to vanish identically at the target trajectory, yielding

$$
\nabla _ { \Theta ^ { [ i ] } } \mathcal { L } \Big | _ { \Theta ^ { [ i ] } = \Theta _ { * } ^ { [ i ] } } = - \mathbb { J } _ { \Theta } ^ { [ i ] ^ { T } } \left( \mathbf { y } - f \mathbb { N } \mathbb { N } _ { \mathrm { d l } } ^ { [ i ] } [ \Theta ^ { [ i ] } ( t ) ] \right) \Big | _ { \Theta ^ { [ i ] } = \Theta _ { * } ^ { [ i ] } } = \mathbf { 0 }\tag{44}
$$

Consequently, guaranteeing the global uniqueness of the instantaneous target $\boldsymbol { \Theta } _ { \ast } ^ { [ i ] } ( t )$ strictly necessitates two core conditions:

C1. The lumped Jacobian must be non-singular, $\mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \neq \mathbf { 0 }$ . This implicitly requires the sign-invariance of $\mathbb { J } _ { \boldsymbol { \theta } } ^ { [ i ] }$ , as a change in sign within the operational domain would topologically lead to $\mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } = \mathbf { 0 }$ via the intermediate value theorem;

C2. The deep mapping $f \mathbb { N } _ { \mathrm { d l } } ^ { [ i ] }$ must be strictly monotonic with respect to $\boldsymbol { \Theta } ^ { [ i ] }$ . This condition is vital to prevent geometric folding or multi-branched overlapping of the parameter-to-output manifold. Given that the global network mapping

$$
f \mathbb { N } \mathbb { N } _ { \mathrm { d l } } ^ { [ i ] } : \quad \mathfrak { M } _ { \Theta ^ { [ i ] } } \to \mathfrak { M } _ { \mathbf { y } ^ { [ n ] } }\tag{45}
$$

is a composition of functions $f \mathbb { N } \mathbb { N } _ { i } , f \mathbb { N } \mathbb { N } _ { i + 1 } , \cdot \cdot \cdot , f \mathbb { N } \mathbb { N } _ { n } .$ , condition C2 is satisfied by ensuring each constitutive layered mapping exhibits strict global monotonicity. Under these constraints, the Euclidean energy surface is guaranteed to preserve a singular, globally attractive unique minimum across the entire domain.

Thus, $\mathcal { F } _ { \boldsymbol { \theta } } ^ { [ i ] }$ as Eqn.(43) and $f \mathbb { N } _ { \mathrm { d l } } ^ { [ i ] }$ as Eqn.(45) are revealed to exhibit an intrinsic structural duality. According to Hadamard-Banach-Mazur theorem [27] for vector fields, the structural constraints imposed on the lumped Jacobian $\mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] }$ and the composition function $f \mathbb { N } _ { \mathrm { d l } } ^ { [ i ] }$ represent two complementary expressions of the same underlying algebraic property. Global convergence therefore dictates that every constituent sub-mapping

$$
f \mathbb { N } \mathbb { N } _ { k } , \quad k \in \{ i , i + 1 , \ldots , n \} ,
$$

preserves strict monotonicity. In a standard artificial neural network, strict monotonicity of the activation functions ensures that their diagonal derivative factors do not change sign. It does not, however, determine the sign of the full lumped Jacobian, which also contains the layer-weight matrices. Condition C1 must therefore be imposed independently of activation monotonicity.

Because the analytical deduction presented above is isomorphic for any arbitrary layer i, global system stability is fundamentally predicated on establishing a global difeomorphism across all sequential layer operators:

$$
f \mathbb { N } \mathbb { N } _ { i } , \quad i \in \{ 1 , 2 , \dots , n \}\tag{46}
$$

The fulfillment of this topological requirement therefore requires both strict monotonicity of the activation functions and the independent non-vanishing, sign-definite Jacobian condition, culminating in the following global convergence theorem.

Theorem (Global Cascaded Convergence and Uniqueness). Consider a deep cascaded neural network online learning system governed by the tracking energy function $\begin{array} { r } { L ( \mathbf { \boldsymbol { \theta } } ) = \frac { 1 } { 2 } \| \mathbf { y } - f \mathbb { N } \mathbb { N } _ { \mathrm { d l } } ( \mathbf { x } , \mathbf { \boldsymbol { \theta } } ) \| ^ { 2 } } \end{array}$ Assume that all hidden-layer activation functions are strictly monotonic and that each associated lumped Jacobian matrix $\mathbb { J } _ { \theta } ^ { [ i ] } , i \in \{ 1 , 2 , . . . , n \}$ , remains non-vanishing and strictly sign-definite across the entire operating domain. Under these topological conditions, the real-time system trajectories converge globally to a unique, time-varying optimal parameter trajectory ${ \boldsymbol { \Theta } } ^ { * } ( t )$

Parameter Learning Parallel Stability. The global cascaded convergence and uniqueness theorem shows that, under active damping, monotonic activation functions, and the Jacobian conditions stated above, the parameter error $\delta \bar { \Theta } ^ { [ i ] }$ converges to zero. This convergence constrains the high-dimensional manifold in Eqn(23) to the lower-dimensional subspace characterized by $\delta \boldsymbol { \Theta } ^ { [ i ] } = \mathbf { 0 }$ . However, as defined by

Eqn(20), the parameter $\boldsymbol { \Theta } ^ { [ i ] }$ is a local composition of vector space spanned by $\boldsymbol { \Theta } _ { k } ^ { [ i ] }$ for $k \in \{ 1 , 2 , \ldots , n _ { i } \}$ Consequently, the manifold Eqn(23) is reduced to

$$
\mathbb { J } _ { \boldsymbol { \Theta } _ { 1 } } ^ { [ i ] } { } ^ { T } \delta \boldsymbol { \Theta } _ { 1 } ^ { [ i ] } + \mathbb { J } _ { \boldsymbol { \Theta } _ { 2 } } ^ { [ i ] } { } ^ { T } \delta \boldsymbol { \Theta } _ { 2 } ^ { [ i ] } + \cdot \cdot \cdot + \mathbb { J } _ { \boldsymbol { \Theta } _ { k } } ^ { [ i ] } { } ^ { T } \delta \boldsymbol { \Theta } _ { k } ^ { [ i ] } + \cdot \cdot \cdot + \mathbb { J } _ { \boldsymbol { \Theta } _ { n _ { i } } } ^ { [ i ] } { } ^ { T } \delta \boldsymbol { \Theta } _ { n _ { i } } ^ { [ i ] } = \mathbf { 0 }\tag{47}
$$

![](images/5fc72a889145e54cc6f4445ad2cdeb1ccaed1834219cf6fef6a365b6d3148a4c.jpg)  
Fig. 4 Topological Isomorphism of Intra-Layer Feedback Learning

As illustrated in Fig.4a, this represents a parallel learning structure wherein the individual learning loops are locally independent and isomorphic. By treating the downstream inference and errorconstruction layers as the physical world seen by the current layer, the independence of the parallel learning becomes apparent. Under the active damping, the convergence and stability of these parallel learning loops are guaranteed, provided the associated activation functions remain monotonic. To formalize this stability, the total energy function of Eqn(41) is expanded as

$$
\mathcal { L } ( \mathbf { y } , \boldsymbol { \Theta } , t ) = \frac { 1 } { 2 } \mathbf { e } _ { y } ^ { T } \mathbf { e } _ { y } + \sum _ { i = 1 } ^ { n } \frac { 1 } { 2 \lambda _ { i } } \sum _ { k = 1 } ^ { n _ { i } } \delta \boldsymbol { \Theta } _ { k } ^ { [ i ] ^ { T } } \delta \boldsymbol { \Theta } _ { k } ^ { [ i ] }\tag{48}
$$

Here, the parallel learning components within a given layer are assumed to share a uniform learning rate, denoted by $\lambda _ { i } .$ . Notably, this uniformity is not a strict requirement. Distinct localized learning rates $\lambda _ { i , k }$ can be implemented without loss of generality. Evaluating the time derivative of the energy function via the same mathematical formalism established for cascaded learning yields

$$
\dot { \mathcal { L } } ( t ) = - \mathbf { e } _ { y } ^ { T } \mathbb { K } _ { y } \mathbf { e } _ { y } - \sum _ { i = 1 } ^ { n } \frac { 1 } { \lambda _ { i } } \sum _ { k = 1 } ^ { n _ { i } } \delta \mathbf { \Theta } _ { k } ^ { [ i ] ^ { T } } \mathbb { K } _ { \mathbf { \Theta } _ { \mathbf { \Theta } _ { k } } } ^ { [ i ] } \delta \mathbf { \Theta } _ { k } ^ { [ i ] } \le 0\tag{49}
$$

According to Lyapunov stability criteria, Eqn(49) implies that $\delta \Theta _ { k } ^ { [ i ] } \ \to \ \mathbf { 0 }$ asymptotically for all $k \in$ $\{ 1 , 2 , \ldots , n _ { i } \}$

Extending this framework to intra-layer parallel learning further confines the parameter space to the manifold defined by

$$
\mathbb { J } _ { \boldsymbol { \Theta } _ { k , 1 } } ^ { [ i ] } \delta \boldsymbol { \Theta } _ { k , 1 } ^ { [ i ] } + \mathbb { J } _ { \boldsymbol { \Theta } _ { k , 2 } } ^ { [ i ] } \delta \boldsymbol { \Theta } _ { k , 2 } ^ { [ i ] } + \cdots + \mathbb { J } _ { \boldsymbol { \Theta } _ { k , l } } ^ { [ i ] } \delta \boldsymbol { \Theta } _ { k , l } ^ { [ i ] } + \cdots + \mathbb { J } _ { \boldsymbol { \Theta } _ { k , n _ { i - 1 } } } ^ { [ i ] } \delta \boldsymbol { \Theta } _ { k , n _ { i - 1 } } ^ { [ i ] } = 0\tag{50}
$$

which represents the localized counterpart to the layer-level manifold in Eqn(47). Because the variables in the above equation are scalars, explicit vector-transpose notation is unnecessary. This formulation establishes parameter convergence at the node level. From a system-dynamics perspective, node-level learning stability can be formalized through an approach isomorphic to the layer-level derivation. To account for these localized parameter estimation errors, the Lyapunov energy candidate of Eqn(48) is expanded as

$$
\mathcal { L } ( \mathbf { y } , \boldsymbol { \Theta } , t ) = \frac { 1 } { 2 } \mathbf { e } _ { y } ^ { T } \mathbf { e } _ { y } + \sum _ { i = 1 } ^ { n } \frac { 1 } { 2 \lambda _ { i } } \sum _ { k = 1 } ^ { n _ { i } } \sum _ { l = 1 } ^ { n _ { i - 1 } } \delta \boldsymbol { \Theta } _ { k , l } ^ { [ i ] } ^ { 2 }\tag{51}
$$

Evaluating the time derivative of this expanded function along the system trajectories yields

$$
\dot { \mathcal { L } } ( t ) = - \mathbf { e } _ { y } ^ { T } \mathbb { K } _ { y } \mathbf { e } _ { y } - \sum _ { i = 1 } ^ { n } \frac { 1 } { \lambda _ { i } } \sum _ { k = 1 } ^ { n _ { i } } \sum _ { l = 1 } ^ { n _ { i - 1 } } \mathbb { K } _ { \boldsymbol { \Theta } _ { k , l } } ^ { [ i ] } \delta \boldsymbol { \Theta } _ { k , l } ^ { [ i ] } \overset { 2 } { \le } 0\tag{52}
$$

Once again, under Lyapunov stability criteria, Eqn(52) guarantees that the localized parameter deviations vanish asymptotically, implying that $\delta \Theta _ { k , l } ^ { [ i ] } \  \ 0$ asymptotically for all $k \in \{ 1 , 2 , \ldots , n _ { i } \}$ and $l \in \{ 1 , 2 , \dots , n _ { i - 1 } \}$

Cross-referencing the error-dynamic manifolds given by Eqn(23) with Eqn(47) and $\operatorname { E q n } ( 5 0 )$ , as well as energy functionals of Eqn(41) with Eqn(48) and Eqn(51), it reveals a fundamental mathematical structural isomorphism of parameter learning across the node, layer, and network levels. Node and layer level learning dynamics correspond structurally to the scalar and vector expansions of the block elements $\mathbb { K } _ { \boldsymbol { \theta } } ^ { [ i ] }$ and $\mathbb { J } _ { \boldsymbol { \theta } } ^ { [ i ] }$ in the state-space matrix A of Eqn(38). Intrinsically, the invariant block structure of A governs global asymptotic stability.

However, while this mathematical proof confirms global stability, it masks the underlying coupled nature of node-level learning. Unlike the layer-level architecture where individual loops operate independently, learning at the node level is inherently coupled. As illustrated in Fig. 4b, this multivariable interdependence is driven simultaneously by both forward information cascades and backward error-propagation pathways.

To reveal the data-driven nature of learning at the node level, the equivalent local learning system is abstracted as follows, in analogy with the layer-level derivation.

$$
\hat { y } ( t ) = f [ \pmb { \theta } ^ { T } ( t ) \mathbf { x } ( t ) ]\tag{53}
$$

where ${ \bf x } ( t )$ is the inputs to the localized learning system at the node-level, and θ the learning parameters as a generalized representation of $\boldsymbol { \Theta } _ { k , l } ^ { [ i ] }$ locally. The output of ${ \hat { y } } ( t )$ shall follow the ideal $y ( t )$ as

$$
y ( t ) = f [ \pmb { \theta } ^ { * T } ( t ) \mathbf { x } ( t ) ]
$$

in which $\pmb \theta ^ { * } ( t )$ is for $\pmb \theta ( t )$ to learn. $f : \mathbb { R } $ R is a strictly monotonic and globally Lipschitz continuous activation function of the node, satisfying

$$
L P _ { \operatorname* { m a x } } \ge f ^ { \prime } ( \chi ) \ge L P _ { \operatorname* { m i n } } > 0
$$

Then, non-linear prediction error is as

$$
e ( t ) = y ( t ) - \hat { y } ( t ) = f [ \pmb { \theta } ^ { * T } ( t ) \mathbf { x } ( t ) ] - f [ \pmb { \theta } ^ { T } ( t ) \mathbf { x } ( t ) ] = \frac { \partial \hat { y } ^ { T } } { \partial \pmb { \theta } } \delta \pmb { \theta } ( t )\tag{54}
$$

The parameter error is defined as $\delta \pmb { \theta } ( t ) = \pmb { \theta } ^ { * } ( t ) - \pmb { \theta } ( t )$ . Defining the energy function as $\begin{array} { r } { \mathcal { L } = \frac { 1 } { 2 } e ^ { 2 } ( t ) } \end{array}$ , the minimum-energy principle gives the localized learning law

$$
\dot { \theta } ( t ) = - \lambda \frac { \partial \mathcal { L } } { \partial \theta } = \lambda e ( t ) \frac { \partial \hat { y } } { \partial \theta } = \lambda \frac { \partial \hat { y } } { \partial \theta } \frac { \partial \hat { y } } { \partial \theta } ^ { T } \delta \theta ( t ) = \lambda [ f ^ { \prime } ( t ) ] ^ { 2 } \mathbf { x } ( t ) \mathbf { x } ^ { T } ( t ) \delta \theta ( t )\tag{55}
$$

This symmetric matrix ${ \mathbf x } ( t ) { \mathbf x } ^ { T } ( t )$ acts as a directional filter for learning. For stability analysis, the positive-definite Lyapunov candidate is

$$
V ( \delta \pmb { \theta } ) = \frac { 1 } { 2 \lambda } \delta \pmb { \theta } ^ { T } ( t ) \delta \pmb { \theta } ( t )
$$

Bearing in mind that $\delta \dot { \pmb \theta } ( t ) = - \dot { \pmb \theta } ( t )$ , the time derivative along the non-linear trajectory becomes

$$
\dot { V } = \frac { 1 } { \lambda } \delta \pmb { \theta } ^ { T } ( t ) \delta \dot { \pmb { \theta } } ( t ) = - [ f ^ { \prime } ( t ) ] ^ { 2 } \delta \pmb { \theta } ^ { T } ( t ) \mathbf { x } ( t ) \mathbf { x } ^ { T } ( t ) \delta \pmb { \theta } ( t )
$$

Since $f ^ { \prime } ( t ) \geq L P _ { \operatorname* { m i n } } > 0$ and the remaining quadratic term is non-negative, we obtain

$$
\begin{array} { r } { \dot { V } \leq - L P _ { \operatorname* { m i n } } ^ { 2 } \parallel \mathbf { x } ^ { T } ( t ) \delta \pmb { \theta } ( t ) \parallel _ { 2 } ^ { 2 } \leq 0 } \end{array}\tag{56}
$$

This guarantees Lyapunov stability, $\delta \pmb \theta ( t )$ is bounded, and the asymptotical solution is

$$
\mathbf { x } ^ { T } ( t ) \delta \pmb { \theta } ( t ) = 0\tag{57}
$$

Since $\begin{array} { r } { e ( t ) = \frac { \partial \hat { y } } { \partial \pmb { \theta } } ^ { T } \delta \pmb { \theta } ( t ) } \end{array}$ , and $\begin{array} { r } { \dot { e } ( t ) = \frac { \partial \hat { y } } { \partial \pmb { \theta } } ^ { T } \delta \dot { \pmb { \theta } } ( t ) } \end{array}$ , then

$$
\dot { \mathcal { L } } = - e ( t ) \frac { \partial \hat { y } } { \partial \pmb { \theta } } ^ { T } \frac { \partial \hat { y } } { \partial \pmb { \theta } } e ( t ) \leq 0
$$

By Barbalat’s Lemma

$$
\operatorname* { l i m } _ { t \to \infty } e ( t ) = 0
$$

While the system efectively minimizes the energy functional ${ \mathcal { L } } ,$ convergence of $\delta \pmb \theta ( t ) = \mathbf 0$ is not guaranteed. In particular, even when the tracking error vanishes $( e  0 )$ , parameter adaptation can stall. Recall that the continuous-time parameter error dynamics evolve as a time-varying linear system governed by

$$
\delta \dot { \pmb { \theta } } ( t ) = - \lambda [ f ^ { \prime } ( t ) ] ^ { 2 } \mathbf { x } ( t ) \mathbf { x } ^ { T } ( t ) \delta \pmb { \theta } ( t )
$$

The matrix $\boldsymbol { \Omega } ( t ) = \mathbf { x } ( t ) \mathbf { x } ^ { T } ( t )$ is an outer product of an $l \times 1$ vector. For any given instant $t ,$ its rank is at most unity. Consequently, for an l-parameter system $( l > 1 )$ , this matrix possesses a high-dimensional null space of dimension l − 1. If the parameter error vector $\delta \pmb \theta ( t )$ projects into the null space of $\Omega ( t )$ , the matrix operation collapses, yielding

$$
\begin{array} { r } { \delta \dot { \pmb \theta } ( t ) = - \lambda \mathbf { x } ( t ) \underbrace { \left[ \mathbf { x } ^ { T } ( t ) \delta \pmb \theta ( t ) \right] } _ { e ( t ) = 0 } = \mathbf { 0 } } \end{array}
$$

Because $\delta \dot { \pmb { \theta } } ( t ) = \mathbf { 0 }$ , the adaptation process completely freezes. The system cannot resolve any parameter errors orthogonal to the instantaneous input vector ${ \bf x } ( t )$ , causing the algorithm to stagnate at an incorrect parameter estimate. Geometrically, the learning process can be interpreted as a trajectory migrating through an l-dimensional parameter error manifold. When the input vector ${ \bf x } ( t )$ lacks suficient variation, the landscape of the energy function L forms a flat-bottomed valley consisting of an afine $l - 1$ subspace. On this sub-manifold, the tracking error is identically zero, $e ( t ) = 0 ,$ yet the parameter error remains non-zero, $\delta \pmb { \theta } ( t ) \neq \mathbf { 0 }$ . The error trajectory quickly descends the steep walls of this valley. However, upon intercepting the floor, tracking error feedback is completely eliminated, permanently trapping the parameter error within this unidentifiable subspace.

To break this null-space trap, the input signal ${ \bf x } ( t )$ must be Persistently Exciting (PE). A vector signal ${ \bf x } ( t )$ is PE if there exist constants $\alpha , T > 0$ such that

$$
\Omega _ { T } ( t ) = \int _ { t } ^ { t + T } \mathbf { x } ( \tau ) \mathbf { x } ^ { T } ( \tau ) d \tau \geq \alpha I\tag{58}
$$

for all $t \geq 0$ . This requires the resulting matrix $\Omega _ { T } ( t )$ to be full rank and strictly positive definite $( \geq \alpha I )$ when the input data are integrated over a moving time window T. The input signal must change direction suficiently to span the entire parameter space. To evaluate how the PE condition resolves the bottleneck for parameter learning convergence, the energy dissipation rate $\dot { V } ( t )$ is integrated over a finite temporal window $[ t , t + T ]$

$$
V ( t + T ) - V ( t ) = \int _ { t } ^ { t + T } { \dot { V } ( \tau ) d \tau } = - \int _ { t } ^ { t + T } \left[ f ^ { \prime } ( \tau ) \right] ^ { 2 } \delta \theta ^ { T } ( \tau ) \Omega ( \tau ) \delta \theta ( \tau ) d \tau
$$

Under the assumption of a suficiently small adaptation gain λ, the parameter error vector evolves slowly relative to the window duration T, permitting the approximation $\delta \pmb \theta ( \tau )$ ≈ $\delta \pmb \theta ( t )$ over the interval. Furthermore, the global difeomorphism requires that the activation derivative remains bounded away from zero, such that $\left[ f ^ { \prime } ( \tau ) \right] ^ { 2 } \geq \beta > 0$ . This separation of timescales allows the parameter error vector to be factored out of the integral operator

$$
V ( t + T ) - V ( t ) \approx - \beta \delta \pmb { \theta } ^ { T } ( t ) \left[ \int _ { t } ^ { t + T } \pmb { \Omega } ( \tau ) d \tau \right] \delta \pmb { \theta } ( t ) = - \beta \delta \pmb { \theta } ^ { T } ( t ) \pmb { \Omega } _ { T } ( t ) \delta \pmb { \theta } ( t )
$$

Substituting the lower bound established by the PE condition, $\Omega _ { T } ( t ) \geq \alpha \mathbf { I }$ , yields the following contractive inequality:

$$
V ( t + T ) - V ( t ) \leq - \alpha \beta \delta { \pmb \theta } ^ { T } ( t ) \delta { \pmb \theta } ( t )
$$

To express this relationship in a standard continuous-time Lyapunov format, the finite-time temporal expectation operator $\mathbb { E } _ { T } [ \cdot ]$ is applied, where $\begin{array} { r } { \mathbb { E } _ { T } [ \dot { V } ( t ) ] = \frac { 1 } { T } \left( V ( t + T ) - V ( t ) \right) } \end{array}$ , translating the inequality into

$$
\mathbb { E } _ { T } [ \dot { V } ( t ) ] \le - \frac { \alpha \beta } { T } \delta \pmb { \theta } ^ { T } ( t ) \delta \pmb { \theta } ( t )\tag{59}
$$

Because $V ( t )$ is lower-bounded by zero and $\mathbb { E } _ { T } [ \dot { V } ( t ) ]$ is strictly negative semi-definite, the integral of the mean dissipation rate remains finite. According to Barbalat’s Lemma, the uniform continuity of the system trajectories ensures that this average dissipation rate asymptotically vanishes. Consequently, the parameter learning error is guaranteed to converge asymptotically to the origin, ensuring exact parameter identification

$$
\operatorname* { l i m } _ { t \to \infty } \mathbb { E } _ { T } [ { \dot { V } } ( t ) ] = 0 \implies \varprojlim _ { t \to \infty } \delta \theta ( t ) = 0 \implies \varprojlim _ { t \to \infty } \theta ( t ) = \theta ^ { * }
$$

This adaptive learning convergence exhibits a close analogy to sliding-mode systems (SMS) [28], where trajectories are constrained to a predefined manifold. Driven by large initial tracking errors, the instantaneous gradient generates an orthogonal restoring force that rapidly forces the parameter error vector onto the time-varying hyper-surface

$$
\mathbf { \boldsymbol { s } } ( t ) = \delta \mathbf { \boldsymbol { \theta } } ^ { T } ( t ) \mathbf { \boldsymbol { x } } ( t ) = \boldsymbol { 0 }\tag{60}
$$

which is the same as $e ( t ) = 0$ according to Eqn(54). However, a fundamental divergence from traditional sliding-mode systems occurs upon interception. Whereas conventional sliding-mode trajectories navigate toward the origin via autonomous internal dynamics, error-feedback gradients vanish identically on $s ( t ) =$ 0, rendering the trajectory instantly stagnant.

PE breaks this stasis by modulating the kernel of $s ( t ) = 0$ continuously. As the regressor vector ${ \bf x } ( t )$ varies continuously under the PE condition, this kernel undergoes a dynamic rotation. Any directional perturbation immediately displaces the parameter state out of the manifold, generating a non-zero tracking error $[ e ( t ) \neq 0 ]$ that reactivates the gradient field and produces a localized contractive force. Topologically, the unique invariant intersection of all these dynamically rotating hyper-planes over any non-vanishing interval contracts exclusively to the origin $\delta \pmb { \theta } ( t ) = \mathbf { 0 }$ . This sliding-mode-like mechanism inherent to data-driven learning is illustrated as Fig.4c, where continuous directional excitation prevents parameter stasis by constraining the trajectory to the zero-tracking manifold.

Theorem (Continuous Online Learning Global Stability). Consider a deep layered neural network of $\hat { \mathbf { y } } = f \mathbb { N } \mathbb { N } _ { \mathrm { d l } } ( \mathbf { x } , \mathbf { \boldsymbol { \Theta } } )$ , where the parameter adaptation is governed by the tracking error feedback $o f \dot { \Theta } ^ { [ i ] } +$ $\mathbb { K } _ { \boldsymbol { \Theta } } ^ { [ i ] } \boldsymbol { \Theta } ^ { [ i ] } = \lambda _ { i } \mathbb { J } _ { \boldsymbol { \Theta } } ^ { [ i ] } \mathbf { e } _ { y }$ . If all hidden-layer activation functions are strictly monotonic, each lumped Jacobian $\mathbb { J } _ { \theta } ^ { [ i ] } , i \in \{ 1 , 2 , . . . , n \}$ , remains non-vanishing and strictly sign-definite across the entire operating domain, and the regressor is persistently exciting in the sense of Eqn (58), the online learning dynamics are asymptotically stable under the Lyapunov energy principle.

## 4 Conclusion

This paper analyses Online Active Learning (OAL) as a stability problem for large-scale neural systems. A decreasing prediction error alone does not establish stable online learning. Multilayer networks admit high-dimensional manifolds of parameter configurations that produce identical outputs. Parameter convergence, cross-layer coupling, and internal zero dynamics must therefore be analysed explicitly.

Four analytical results are established. First, under bounded external variation, the OAL error feedback law makes the prediction-error energy non-increasing, as shown in Eqn(15). Second, the block state-space representation in $\operatorname { E q n } ( 2 8 )$ shows no direct parameter-to-parameter coupling between diferent layers. Their interaction is mediated by the shared error $\mathbf { e } _ { y } .$ . These cross-layer terms are skew-symmetric and therefore energy-conservative. Spectral analysis shows that the outer tracking loop is damped, whereas the internal zero dynamics remain marginally stable in $\ker ( \mathbb { J } _ { s k } )$

Third, the analysis identifies persistency of excitation and active damping as two mechanisms for removing this marginal stability. Under active damping, the cascaded Lyapunov analysis in $\mathrm { E q n } ( 4 2 )$ gives $\mathbf { e } _ { y }  \mathbf { 0 }$ and $\delta \Theta ^ { [ i ] } \to \mathbf { 0 }$ for every layer. The isomorphic argument extends to intra-layer and nodelevel parameters, where the PE condition contracts the zero-tracking manifold onto the origin. Fourth, uniqueness of the time-varying target trajectory ${ \boldsymbol { \theta } } ^ { * } ( t )$ requires difeomorphic layer mappings. For the neural architecture considered here, this requires strictly monotonic activation functions together with non-vanishing, sign-definite lumped Jacobians. In the simplified damping case, the parameter energy decays as $e ^ { - 2 \kappa t }$ , so κ sets the exponential decay rate.

These results are analytical and require experimental validation. Future work will run the primary framework of Fig. 1b on continuously arriving sensory streams. The experiments will measure prediction-error decay and, where an identifiable reference is available, parameter-deviation decay during uninterrupted online operation. They will also test whether natural streams satisfy Eqn(36) and quantify how κ balances retention of learned structure against adaptation speed. Comparisons with ofline-trained models of the same capacity will assess whether monotonic activations, suficient excitation, and appropriate damping are achieved in practical vision-based perception systems for autonomous driving and robotics.

## References

[1] Rumelhart, D., Hinton, G. & Williams, R. Learning representations by back-propagating errors. Nature 323, 533–536 (1986).

[2] LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436–444 (2015).

[3] Hinton, G. The forward-forward algorithm: Some preliminary investigations. arXiv preprint arXiv:2212.13345 (2022).

[4] Zador, A. M. A critique of pure learning and what artificial neural networks can learn from animal brains. Nature communications 10, 3770 (2019).

[5] De Lange, M. et al. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 3366–3385 (2021).

[6] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C. & Wermter, S. Continual lifelong learning with neural networks: A review. Neural Networks 113, 54–71 (2019).

[7] Wang, D., Shelhamer, E., Liu, S., Olshausen, B. & Darrell, T. Tent: Fully test-time adaptation by entropy minimization. International Conference on Learning Representations (2021).

[8] Zhou, D.-X. Universality of deep convolutional neural networks. Applied and Computational Harmonic Analysis 48, 787–794 (2020).

[9] Amjad, R. A. & Geiger, B. C. Learning representations for neural network-based classification using the information bottleneck principle. IEEE Transactions on Pattern Analysis and Machine Intelligence 42, 2225–2239 (2019).

[10] <sup>˚</sup>Astr¨om, K. J. Adaptive control (Springer, 1995).

[11] Ioannou, P. A. & Sun, J. Robust adaptive control (Prentice-Hall, 1996).

[12] Slotine, J.-J. E., Li, W. et al. Applied nonlinear control (Prentice hall, 1991).

[13] Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J. & Hinton, G. Backpropagation and the brain. Nature Reviews Neuroscience 21, 335–346 (2020).

[14] Rao, R. & Ballard, D. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field efects. Nature Neuroscience 2, 79–87 (1999).

[15] Friston, K. & Kiebel, S. Predictive coding under the free-energy principle. Philosophical Transactions of the Royal Society B: Biological Sciences 364, 1211–1221 (2009).

[16] Hosoya, T., Baccus, S. & Meister, M. Dynamic predictive coding by the retina. Nature 436, 71–77 (2005).

[17] Huang, Y. & Rao, R. Predictive coding. Wiley Interdisciplinary Reviews: Cognitive Science 2, 580–593 (2011).

[18] Knill, D. C. & Pouget, A. The bayesian brain: the role of uncertainty in neural coding and computation. TRENDS in Neurosciences 27, 712–719 (2004).

[19] C¸ atal, O., Wauthier, S., De Boom, C., Verbelen, T. & Dhoedt, B. Learning generative state space models for active inference. Frontiers in Computational Neuroscience 14, 574372 (2020).

[20] Kirchhof, M., Parr, T., Palacios, E., Friston, K. & Kiverstein, J. The markov blankets of life: autonomy, active inference and the free energy principle. Journal of The royal society interface 15, 20170792 (2018).

[21] Isidori, A. The zero dynamics of a nonlinear system: From the origin to the latest progresses of a long successful story. European Journal of Control 19, 369–378 (2013).

[22] Qian, N. On the momentum term in gradient descent learning algorithms. Neural networks 12, 145–151 (1999).

[23] Farkas, B. & Wegner, S.-A. Variations on barb˘alat’s lemma. The American Mathematical Monthly 123, 825–830 (2016).

[24] Nguyen, N. T. Lyapunov stability theory (Springer, 2018).

[25] Barkana, I. Defending the beauty of the invariance principle. International Journal of Control 87, 186–206 (2014).

[26] Friston, K. The free-energy principle: a unified brain theory? Nature Reviews Neuroscience 11, 127–138 (2010).

[27] Plastock, R. Homeomorphisms between banach spaces. Transactions of the American mathematical society 200, 169–183 (1974).

[28] Utkin, V., Guldner, J. & Shi, J. Sliding mode control in electro-mechanical systems (CRC press, 2017).