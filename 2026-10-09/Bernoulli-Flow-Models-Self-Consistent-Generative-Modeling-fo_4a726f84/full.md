# Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data

Hao Mo<sup>1</sup>, Liying Yang<sup>1</sup>, Shumin Yao<sup>2</sup>, Xinxing Yu<sup>1</sup>, Ajian Liu<sup>3†</sup>, Xudong Mao<sup>4</sup>, Yanyan Liang<sup>1†</sup> <sup>1</sup>SCSE, Macau University of Science and Technology   
<sup>2</sup>Department of Broadband Communication, Pengcheng Laboratory   
<sup>3</sup>Institute of Automation, Chinese Academy of Sciences (CASIA) <sup>4</sup>School of Artificial Intelligence, Sun Yat-sen University <sup>†</sup>Corresponding authors

## Abstract

Binary diffusion models typically require a massive number of function evaluations (NFEs) to generate high-quality samples, making practical inference computationally expensive. Reducing NFEs while preserving sample quality without relying on distillation or additional training remains a significant challenge. However, existing binary diffusion models sequentially define a discrete one-step forward path and subsequently derive the reverse posterior. In low-NFE scenarios that require cross-step sampling, these models incorrectly approximate the true multi-step likelihood using a single-step likelihood transition, which severely degrades sample quality. To address this fundamental limitation and completely decouple the generative dynamics from fixed discrete time steps, we propose Bernoulli Flow Models (BFM). Rather than building upon sequential one-step Markov diffusion chains, we predefine a unified continuous global Bernoulli probability flow path between data distributions and pure noise, from which we derive analytical closed-form posterior transitions over arbitrary time intervals. Consequently, reducing the inference NFE in BFM is no longer an approximation of skipped discrete steps; it simply requires re-evaluating the analytical posterior on the new time intervals. This eliminates the structural training-inference mismatch inherent to discrete chains, yielding strictly self-consistent low-NFE sampling. Experimental results show that BFM is highly robust to aggressive NFE reduction. On the LSUN Churches 256x256 dataset, a 256-step-trained BFM achieves an FID of 9.22 when sampled with only 16 steps, whereas the state-of-the-art discrete baseline severely degrades to 204.10. BFM also remains competitive with both continuous and discrete generative baselines under standard full-step inference. Ultimately, these results establish BFM as a theoretically rigorous, self-consistent, and practically effective framework for fast binary data generation.

## 1 Introduction

Generative modeling over binary variables is a fundamental problem in binary image modeling, binarized latent representation learning, and statistical physics systems with two-state configurations. Recent binary generative models have shown that diffusion-style methods can achieve strong sample quality in binary spaces, including bit-based continuous diffusion formulations [4], Binary Latent Diffusion (BLD) [32], and recent binary diffusion models [18, 34]. However, these models still typically require hundreds or even thousands of sequential function evaluations (NFEs) to generate high-fidelity samples, making practical inference computationally expensive.

A natural objective is to reduce the inference NFE while preserving sample quality without additional distillation, consistency training, or task-specific retraining. In continuous diffusion models, this problem has been extensively studied through training-based acceleration, such as progressive distillation [25] and consistency models [29], as well as training-free samplers, such as DDIM [28] and DPM-Solver [21], which evaluate the same continuous-time dynamics on a coarser inference grid [13, 30, 19]. In binary state spaces, however, such training-free acceleration is far less straightforward, because many binary diffusion models are defined through fixed one-step discrete Markov transitions rather than an arbitrary-interval continuous path.

The key difficulty is that low-NFE binary sampling is not merely a matter of skipping steps. Under the original full-step inference schedule, the reverse process remains aligned with the training-time Markov chain. Once the inference grid is coarsened, each reverse update must jump across multiple intermediate training intervals. For example, sampling a 256-step-trained binary model with only 16 steps requires each inference step to represent a large cross-step transition. A valid low-NFE sampler therefore needs a reverse posterior defined over arbitrary time intervals; reusing the original one-step posterior for such enlarged intervals no longer describes the same generative process.

This exposes a fundamental self-consistency issue in existing binary diffusion samplers. Although multi-step transitions may be derived in principle for a discrete Markov chain, practical binary diffusion samplers are often tied to the local transition rule used during training. When the sampling schedule changes after training, the required cross-step posterior is approximated by local one-step likelihood transitions, introducing a structural mismatch between the training-time process and the low-NFE inference-time process. As the step budget becomes smaller, this mismatch is amplified, leading to severe degradation in sample quality. Thus, the central obstacle to fast binary generation is not simply the lack of a faster numerical solver, but the lack of an analytical posterior mechanism that remains valid under changed inference grids.

Recent continuous-time discrete denoising and discrete flow-matching frameworks provide important foundations for discrete generative modeling [2, 31, 20, 3, 10, 6, 7]. Nevertheless, these general formulations do not directly resolve the Bernoulli-specific low-NFE problem considered here: deriving closed-form posterior transitions over arbitrary time intervals so that changing the inference NFE becomes a rigorous reparameterization of the same binary generative process, rather than an approximate cross-step sampler.

To address this limitation, we propose Bernoulli Flow Models (BFM), a self-consistent generative framework for binary data. Instead of defining generation through fixed one-step Markov transitions, BFM starts from a unified continuous Bernoulli probability path between data and pure noise. From this path, we derive closed-form marginal, transition, and posterior distributions over arbitrary time intervals. Consequently, reducing the number of inference steps only requires re-evaluating the same analytical posterior on the new time intervals, which decouples the generative dynamics from a fixed training-time discretization and makes low-NFE sampling self-consistent by construction. Our contributions are summarized as follows: Our contributions can be summarized as follows:

• A Closed-Form Bernoulli Generative Framework: We propose Bernoulli Flow Models (BFM), which predefine a continuous Bernoulli probability path and derive analytical closed-form marginal, transition, and posterior distributions from this path.

• Self-Consistent Low-NFE Inference: The resulting arbitrary-interval posterior enables principled cross-step updates when the inference NFE is reduced after training. Instead of approximating a large multi-step jump with local one-step rules, BFM recomputes the analytical posterior on the new time intervals, yielding self-consistent low-NFE sampling.

• Effective Training-Free Acceleration in Practice: Experiments show that BFM is highly robust to aggressive NFE reduction. From a single 256-step checkpoint on LSUN Churches (256 × 256), BFM maintains an FID of 9.22 with only 16 sampling steps, whereas the corresponding baseline degrades to 204.10. Additional results show that BFM remains competitive under the original full-step inference setting across diverse tasks.

## 2 Related Work

Binary and Bit-Level Flow/Diffusion Models. Binary generative modeling has recently received increasing attention. Early work already explored diffusion-style modeling on binary state spaces [27]. Analog Bits [4] proposed a bit-encoded continuous diffusion approach, where discrete variables are represented as binary bits and then modeled as real-valued analog bits by a continuous diffusion model. Binary Latent Diffusion (BLD) [32] more directly studied diffusion-based generation in binarized latent spaces and demonstrated strong high-resolution image generation performance. Other recent works further investigated Bernoulli or binary diffusion formulations, such as masked Bernoulli diffusion for anomaly detection [34] and Binary Diffusion Probabilistic Models (BDPM) [18]. Concurrently, Binary Flow Matching [14] studies robust flow matching on binary manifolds from the perspective of prediction–loss space alignment, showing that aligning the training objective with the signal space can remove singular weighting and improve training stability. These works demonstrate the promise of binary or bit-level diffusion/flow modeling. However, their focus is mainly on representation, training objectives, or binary-space generative quality, while the problem of self-consistent low-NFE inference from a fixed checkpoint remains underexplored.

Fast Sampling and Training-Free Acceleration. A major limitation of diffusion-based generative models is their slow iterative sampling, which typically requires many sequential function evaluations (NFEs). In continuous domains, extensive work has studied acceleration. Training-based methods, such as progressive distillation [25] and consistency models [29], reduce sampling steps through an additional training stage. Training-free samplers, such as DDIM [28] and DPM-Solver [21], instead accelerate sampling directly from a pretrained model by evaluating the reverse dynamics on a coarser inference grid. These techniques have made low-NFE generation practical for continuous diffusion models. However, binary diffusion models cannot directly rely on continuous-domain ODE or SDE solvers, especially when the reverse process is defined through discrete Bernoulli transitions. Reducing the NFE then requires valid cross-step posterior updates over enlarged time intervals, rather than simply applying a continuous-domain solver. This makes training-free acceleration in binary generative modeling substantially less developed.

Relation to General Discrete Diffusion and Flow Models. A separate line of work studies generative modeling over broader discrete or categorical state spaces, including discrete diffusion models [1], continuous-time discrete denoising frameworks [2, 31, 20], and recent discrete flowmatching formulations [3, 11, 6, 7]. These works provide important general foundations for discrete generative modeling. Our work is complementary but more specialized: rather than proposing a generic categorical diffusion or flow construction, we focus on Bernoulli binary variables and derive closed-form posterior transitions over arbitrary time intervals, enabling self-consistent low-NFE inference from a single pretrained checkpoint.

## 3 Bernoulli Flow Models

In this section, we present the theoretical framework of Bernoulli Flow Models (BFM). We detail how a unified continuous global Bernoulli probability flow path is established to completely decouple the generative dynamics from fixed discrete time steps, and formally derive the resulting analytical probability distributions.

## 3.1 Notation and Problem Formulation

Consider an arbitrary binary data distribution $p \mathbf { x } ( \mathbf { x } )$ over the sample space $\mathcal { X } = \{ 0 , 1 \} ^ { d }$ , where $d \in \mathbb { N } ^ { + }$ denotes the dimensionality, and X is a discrete random vector. We denote the Bernoulli distribution as B.

Given target dataset observations $X _ { 0 } \sim \pi _ { 0 }$ and a prior $X _ { 1 } \sim \pi _ { 1 }$ (where $\pi _ { 1 } = B ( 0 . 5 )$ represents binary white noise), we define an optimal transport-style Bernoulli probability path. Analogous to linear interpolation in continuous spaces, we construct the path directly in the Bernoulli parameter space across continuous time $t _ { \tau } \in [ 1 , 0 ]$

$$
X _ { t _ { \tau } } \sim \mathcal { B } ( t _ { \tau } X _ { 0 } + ( 1 - t _ { \tau } ) X _ { 1 } ) , \quad \tau = 0 , 1 , \dots , T ,\tag{1}
$$

where $X _ { t _ { 0 } } \equiv X _ { 0 }$ and $X _ { t _ { T } } \equiv X _ { 1 }$ . The random variable $X _ { t _ { \tau } }$ strictly follows the Bernoulli probability flow path from the data to the prior.

## 3.2 Bernoulli Probability Flow Dynamics

Building upon the optimal transport-style Bernoulli probability path defined above, and assuming the prior distribution $\pi _ { 1 }$ is a uniform Bernoulli distribution B(0.5) (representing binary white noise), the

marginal transition probability from the target observation $X _ { 0 }$ (or $X _ { t _ { 0 } } )$ to an arbitrary intermediate state $X _ { t }$ can be analytically expressed as:

$$
\begin{array} { c } { { q ( X _ { t _ { \tau } } | X _ { t _ { 0 } } ) = B ( X _ { t _ { \tau } } ; \alpha _ { t _ { \tau } } ) , } } \\ { { \alpha _ { t _ { \tau } } = g \left( X _ { t _ { 0 } } , \displaystyle \frac { 1 - t _ { \tau } } { 2 } \right) , } } \\ { { g ( a , b ) = a ( 1 - b ) + b ( 1 - a ) . } } \end{array}\tag{2}
$$

By exploiting the Markov property along the constructed path, we can further derive the one-step forward transition probability between sequential states $X _ { t _ { \tau - 1 } }$ and $X _ { t _ { \tau } }$

$$
\begin{array} { c } { { q ( X _ { t _ { \tau } } | X _ { t _ { \tau - 1 } } ) = B ( X _ { t \tau } ; g ( X _ { t _ { \tau - 1 } } , \gamma _ { t _ { \tau } } ) ) , } } \\ { { \gamma _ { t _ { \tau } } = \frac { 0 . 5 ( t _ { \tau - 1 } - t _ { \tau } ) } { t _ { \tau - 1 } } , } } \end{array}\tag{3}
$$

where $\gamma _ { t . }$ denotes the explicit flip probability from $X _ { t _ { \tau - 1 } } \mathrm { ~ t o ~ } X _ { t _ { \tau } }$

Using Bayes’ theorem, the ground-truth posterior distribution of the previous state $X _ { t _ { \tau - 1 } }$ conditioned on the current state $X _ { t _ { \tau } }$ and the source data $X _ { t _ { 0 } }$ is formulated as:

$$
q ( X _ { t _ { \tau - 1 } } | X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = \frac { q ( X _ { t _ { \tau } } | X _ { t _ { \tau - 1 } } , X _ { t _ { 0 } } ) q ( X _ { t _ { \tau - 1 } } | X _ { t _ { 0 } } ) } { q ( X _ { t _ { \tau } } | X _ { t _ { \tau - 1 } } ) q ( X _ { t _ { \tau - 1 } } | X _ { t _ { 0 } } ) + q ( X _ { t _ { \tau } } | \neg X _ { t _ { \tau - 1 } } ) q ( \neg X _ { t _ { \tau - 1 } } | X _ { t _ { 0 } } ) } ,\tag{4}
$$

which simplifies into the analytical closed-form solution:

$$
q ( X _ { t _ { \tau - 1 } } | X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = \mathcal { B } \left( X _ { t _ { \tau - 1 } } ; \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) \alpha _ { t _ { \tau - 1 } } } { g ( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) , 1 - \alpha _ { t _ { \tau - 1 } } ) } \right) .\tag{5}
$$

Similar to the $x _ { 0 }$ -prediction objective in continuous DDPM [13], we parameterize our generative flow model $f _ { \theta }$ to directly predict the uncorrupted data $X _ { t _ { 0 } }$ . This design aligns the model output with the analytical Bernoulli posterior probability path, thereby enabling the generative process to sample back to the target distribution $\pi _ { 0 }$ from the prior $\pi _ { 1 }$ . The backward Bernoulli flow transition step is parameterized as:

$$
p _ { \theta } ( X _ { t _ { \tau - 1 } } | X _ { t _ { \tau } } ) = \mathcal { B } \left( X _ { t _ { \tau - 1 } } ; \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) g \left( f _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } ) , \frac { 1 - t _ { \tau - 1 } } { 2 } \right) } { g \left( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) , 1 - g \left( f _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } ) , \frac { 1 - t _ { \tau - 1 } } { 2 } \right) \right) } \right) .\tag{6}
$$

Here, $f _ { \theta }$ is parameterized to directly predict the uncorrupted data $X _ { t _ { 0 } }$ . Following the standard $x _ { 0 } .$ -prediction paradigm widely adopted in diffusion and flow matching models, we employ binary cross-entropy (BCE) to formulate the training objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { B C E } } ( \theta ) : = - \mathbb { E } _ { t _ { \tau } , X _ { t _ { 0 } } , X _ { t _ { \tau } } } \left[ X _ { t _ { 0 } } \log \left( f _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } ) \right) \right. } \\ { \left. + \left( 1 - X _ { t _ { 0 } } \right) \log \left( 1 - f _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } ) \right) \right] . } \end{array}\tag{7}
$$

Detailed proofs for the derivations in this section are provided in Appendix A.

## 3.3 Self-Consistent Cross-Step Sampling

A fundamental advantage of BFM emerges during fast inference when the number of function evaluations (NFEs) is aggressively reduced. Suppose we change the inference time grid and need to jump from $t _ { \tau }$ directly to a distant step $t _ { \tau - k } \ ( k > 1 )$ . In existing heuristic chain-based models, navigating this large stride requires accumulating multiple local one-step transitions, leading to severe approximation errors.

In contrast, BFM naturally supports analytical cross-step posterior sampling. Because the marginal parameters α and the transition flip probabilities $\gamma$ in our framework are continuous analytical functions of time $t \in [ 1 , 0 ]$ , we can directly compute the analytical transition over any arbitrary interval. By generalizing Eq. 3, the analytical cross-step forward transition from $t _ { \tau - k } \tan t _ { \tau }$ is uniquely determined by:

$$
\gamma _ { t _ { \tau } | t _ { \tau - k } } = \frac { 0 . 5 ( t _ { \tau - k } - t _ { \tau } ) } { t _ { \tau - k } } .\tag{8}
$$

By substituting $\gamma _ { t _ { \tau } | t _ { \tau - k } }$ and the corresponding marginal $\alpha _ { t _ { \tau - k } }$ into Bayes’ theorem, we establish the following core theoretical guarantee of our framework:

Theorem 1 (Analytical Cross-Step Posterior). For any continuous time steps $t _ { \tau - k }$ and $t _ { \tau }$ where $k > 1 _ { : }$ , the analytical cross-step posterior ofthe Bernoulli Flow Model admits a perfect closed-form solution:

$$
q ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = \mathcal { B } \left( X _ { t _ { \tau - k } } ; \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) \alpha _ { t _ { \tau - k } } } { g ( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) , 1 - \alpha _ { t _ { \tau - k } } ) } \right) .\tag{9}
$$

Proof. The rigorous derivation of Theorem 1 is provided in Appendix A.

During fast sampling, we simply re-parameterize the linear interpolation of t on [1, 0] to the new NFE budget K (where $K \ll T )$ . We refer to this property as the self-consistency of BFM under cross-step sampling. It mathematically guarantees that BFM eliminates the structural transition errors that plague heuristic diffusion models at low NFEs, fully explaining its remarkable empirical robustness.

## 3.4 Model Training and Sampling

Algorithm 1 Training procedure.   
1: Given: Binary diffusion model $f _ { \theta }$ parametrized by T ; A dataset X.   
2: Given: Diffusion steps $T ;$ Probability path defined by eq. (1); Training steps I.   
3: Initializing $\mathcal { T } _ { \theta }$   
4: for Step $i = 1 : I$ do   
5: Sampling data $\mathbf { x } _ { t _ { 0 } } \sim \mathbf { X } .$ , and time step $\tau \sim \{ 1 , \dots , T \}$   
6: Obtaining $\mathbf { x } _ { t }$ using $\mathbf { x } _ { t _ { 0 } } , t _ { \tau }$ , and probability path with $\mathsf { e q . } ( 1 ) .$   
7: Predicting the probability that the state is $\mathbf { x } _ { t _ { 0 } }$ using $f _ { \theta } ( \dot { \mathbf { x } _ { t _ { \tau } } } , t _ { \tau } ) = \sigma ( \mathcal { T } _ { \theta } ( \mathbf { x } _ { t _ { \tau } } , t _ { \tau } ) / \kappa )$   
8: Calculating loss $\mathcal { L }$ using eq. (7).   
9: Backpropagating L and updating θ.   
10: end for   
11: Return Binary diffusion model $f _ { \theta } .$   
Algorithm 2 Sampling procedure.   
1: Given: Trained binary diffusion model f ;The sample shape dimension shape.   
2: Given: Diffusion steps $T ;$ Probability path defined by eq. (1); Temperature κ.   
3: Sampling $\mathbf { x } _ { t _ { T } } =$ Bernoull $\left( \mathbf { x } ^ { \mathrm { i n i t } } \right)$ , where $\mathbf { x } ^ { \mathrm { i n i t } } \in \mathbb { R } ^ { s h a \bar { p } e }$ and contains 0.5 only.   
4: for Step $\tau = T$ : 1 do   
5: Predicting $p _ { \theta } ( \mathbf { x } _ { t _ { \tau - 1 } } )$ with $f _ { \theta } ( \mathbf { x } _ { t _ { \tau } } , t _ { \tau } ) = \sigma ( \mathcal { T } _ { \theta } ( \mathbf { x } _ { t _ { \tau } } , t _ { \tau } ) / \kappa )$ and eq. (6).   
6: Sampling $\mathbf { x } _ { t _ { \tau - 1 } } =$ Bernoulli $\left( p _ { \theta } ( \mathbf { x } _ { t _ { \tau - 1 } } ) \right)$   
7: end for   
8: $\hat { \mathbf { x } } _ { t _ { 0 } } = \mathbf { x } _ { t _ { \tau - 1 } }$   
9: Return the final sample $\hat { \mathbf { x } } _ { t _ { 0 } } .$

We now describe how BFM can be trained and sampled by minimal modification of existing binary diffusion [32] training and sampling architectures, as summarized in Algorithms 1 and 2, where we highlight the differences between BFM and existing binary diffusion models in blue. The prediction of X<sub>t</sub> $X _ { t _ { 0 } }$ is implemented via a neural network $f _ { \theta } ( \bar { X } _ { t _ { \tau } } , t _ { \tau } \bar { ) } = \sigma ( \tau _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } ) / \kappa )$ , where $\sigma ( \cdot )$ is the sigmoid function, $\mathcal { T } _ { \theta }$ is a neural network with parameters θ, and κ is a temperature hyperparameter. The temperature is used to control the diversity of the generated samples. A smaller temperature leads to less diverse samples, while a larger temperature increases diversity.

## 4 Experiments

We evaluate BFM from four perspectives. First, we validate our central claim that BFM maintains stability under training-free NFE reduction using a single pretrained checkpoint. Second, we verify its competitiveness on high-dimensional latent image benchmarks under standard sampling. Third, we demonstrate direct generative performance on binary manifolds via Binarized MNIST. Fourth, we conduct an ablation study to analyze the impact of different probability path schedulers. Furthermore, Appendix C details BFM’s applicability beyond image synthesis by modeling the Ising binary physical system.

## 4.1 Experimental Setup

To ensure a rigorous and fair comparison with BLD [32], we implement BFM directly on top of the official BLD codebase. We fully follow the experimental settings of BLD, using the identical full transformer backbone and hyperparameters. All experiments are conducted on two NVIDIA GeForce RTX 4090 GPUs. We use the Adam optimizer with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9$ , and $\epsilon = 1 \times 1 0 ^ { - 8 }$ together with a warm-up scheduler whose peak learning rate is $2 \times 1 0 ^ { - 4 }$ after 10K iterations. Unless otherwise specified, we set the temperature κ to 0.9 as in the default BLD setup, and compute all FID, Precision, and Recall metrics from 50K generated samples compared against the corresponding training datasets.

## 4.2 Self-Consistent Low-NFE Sampling

We first evaluate the main practical regime targeted by our method: training-free acceleration, where the number of function evaluations (NFEs) is reduced at inference time without retraining or distillation. This setting directly tests whether BFM remains stable when the inference grid is coarsened after training.

To this end, we compare BFM and BLD on LSUN Churches 256 × 256 under a fixed-checkpoint protocol. Specifically, both models are trained with 256 sampling steps using the linear probability path scheduler, and we evaluate the 100K-iteration checkpoints while varying only the inference NFE. All other training and evaluation settings follow Sec. 4.1.

![](images/27f92efa24abc19cdc74db3a96fb0429d8fba58c4be75552bfb3fe8262180cc1.jpg)  
(a) Low-NFE sampling robustness.

![](images/24d9aa75ebbea3daef9e63bad3443e36f210291e0724993fdc3aaeaf197aeda4.jpg)  
(b) Training dynamics.  
Figure 1: Comparison of BFM and BLD under reduced inference steps and during training. (a) FID versus NFE on LSUN Churches $( 2 5 6 \times 2 5 6 )$ , evaluated from the 100K-iteration checkpoints trained with 256 sampling steps using the linear probability path scheduler, where only the inference NFE is varied. (b) Training convergence comparison on LSUN Bedrooms $( 2 5 6 \times 2 5 6 )$ . We visualize the FID scores from 100K to 400K training iterations $( \times 1 0 ^ { 3 } )$ . Both BFM and BLD are trained under identical experimental settings using a linear scheduler. BFM reaches lower FID consistently throughout this range.

Fig. 1a reports the core low-NFE result. Starting from the same 256-step training setup, BFM remains highly stable as the inference NFE is reduced, whereas BLD deteriorates rapidly under aggressive NFE reduction. In particular, when sampled with only 16 inference steps, BFM achieves an FID of 9.22 under the linear path scheduler, while the discrete baseline BLD severely degrades to 204.10. This result directly supports our central claim that BFM is substantially more robust than chain-based Bernoulli diffusion under training-free NFE reduction.

Fig. 1b further compares the training dynamics from 100K to 400K iterations on LSUN Bedrooms $2 5 6 \times 2 5 6$ . Under identical architectures, schedulers, and optimization settings, BFM consistently reaches lower FID values than BLD throughout this training range. This behavior is consistent with the structural design of BFM. In our framework, the supervision at different sampled time points is induced by the same closed-form Bernoulli probability path, so different sampled times correspond to different intervals of one unified transition family. In contrast, BLD is defined on a fixed one-step discrete chain, where reverse transitions are tied to specific grid steps. This path-consistent parameterization gives BFM a more coherent training signal across time, which empirically translates into faster FID reduction during training.

To make this behavior visually explicit, Fig. 2 compares samples generated from the respective 256-step checkpoints of BLD and BFM under different inference NFEs. As the NFE decreases, BLD quickly becomes blurry and structurally unstable, while BFM preserves global scene structure and semantic layout across a broad range of NFEs. This qualitative evidence is fully consistent with the quantitative FID curve.

![](images/271caf26d3bd75e8f50af42ffbb027d22fd4a0d2e5a707d7b482eb7853ee7943.jpg)  
(a) BLD

![](images/1760f761c9e31519f2563aafff18f4f8d5c663f0103f0cd54284fd9bf9afaed0.jpg)  
(b) BFM  
Figure 2: Visual comparison of samples generated with different NFEs from the respective checkpoints of BLD and BFM, both trained with 256 sampling steps on LSUN Churches 256 × 256. For each method, each row corresponds to a different NFE setting, while images in the same column share the same initial latent code.

The practical significance of these results is that BFM supports training-free acceleration from a single pretrained checkpoint. Reducing the NFE does not require retraining, distillation, or switching to a different sampler. Instead, the model remains stable because the reverse updates can be recomputed on the new coarser inference grid. A more detailed discussion of the corresponding cross-step posterior discrepancy in BLD and the self-consistency mechanism of BFM is deferred to Appendix D.

## 4.3 Standard Sampling Performance on High-Dimensional Image Generation

We next evaluate BFM under the standard sampling setting to examine its generation quality with the original inference budget. This experiment complements the low-NFE analysis and verifies that BFM remains competitive in the conventional full-step sampling regime.

To evaluate BFM on challenging image synthesis tasks, we use it as a latent-space generator. Following BLD [32], we employ the pretrained encoder and decoder to map images from LSUN Bedrooms and Churches [35], and FFHQ [16] into a binary latent space of size $1 6 \times 1 6 \times 6 4$ . Detailed implementation specifics of the autoencoder can be found in [32].

A direct comparison to the originally reported BLD numbers is not fully controlled, because the corresponding training checkpoints are not publicly available and the original training environment differs from ours. Therefore, instead of mixing unmatched results in the main comparison, we reproduce BLD using its official codebase under the same machine environment and the same training configuration as BFM, including the backbone, optimizer, learning rate, loss function, and iteration budget. Table 1 reports only this unified comparison. Under this strictly matched setting, BFM consistently outperforms the reproduced BLD across all three datasets (e.g., FID 5.86 vs. 6.22 on LSUN Bedrooms and 10.87 vs. 11.46 on FFHQ).

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Steps</td><td colspan="3">LSUN-Bedrooms 256x256</td><td colspan="3">LSUN-Churches 256x256</td><td colspan="3">FFHQ 256x256</td></tr><tr><td>FID ↓</td><td>Prec. ↑</td><td>Recall ↑</td><td>FID ↓</td><td>Prec. ↑</td><td>Recall ↑</td><td>FID↓</td><td>Prec. ↑</td><td>Recall ↑</td></tr><tr><td>Continuous Generative Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>StyleGAN [16, 17]</td><td></td><td>2.35</td><td>0.59</td><td>0.48</td><td>3.86</td><td>0.60</td><td>0.43</td><td>4.16</td><td>0.71</td><td>0.46</td></tr><tr><td>LDM-4/8/4 [24]</td><td>200</td><td>2.95</td><td>0.66</td><td>0.48</td><td>4.02</td><td>0.64</td><td>0.52</td><td>4.98</td><td>0.73</td><td>0.50</td></tr><tr><td>Patch-DM [9]</td><td>50</td><td>6.04</td><td>0.56</td><td>0.44</td><td>5.49</td><td>0.62</td><td>0.53</td><td>10.02</td><td>0.68</td><td>0.44</td></tr><tr><td>VQ-LCMD [22]</td><td>200</td><td>4.16</td><td>0.72</td><td>0.40</td><td>4.99</td><td>0.75</td><td>0.42</td><td>7.25</td><td>0.72</td><td>0.46</td></tr><tr><td>Discrete Generative Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>D3PM [1]</td><td>200</td><td>6.60</td><td>0.60</td><td>0.35</td><td>6.02</td><td>0.68</td><td>0.39</td><td>9.49</td><td>0.71</td><td>0.41</td></tr><tr><td>VQ-Diffusion [12]</td><td>200</td><td>7.19</td><td>0.54</td><td>0.37</td><td>6.88</td><td>0.72</td><td>0.37</td><td>8.79</td><td>0.70</td><td>0.43</td></tr><tr><td>BLD* [32]</td><td>64</td><td>6.22</td><td>0.69</td><td>0.36</td><td>5.48</td><td>0.66</td><td>0.41</td><td>11.46</td><td>0.68</td><td>0.48</td></tr><tr><td>BFM (ours)</td><td>64</td><td>5.86</td><td>0.69</td><td>0.37</td><td>5.32</td><td>0.66</td><td>0.41</td><td>10.87</td><td>0.68</td><td>0.49</td></tr></table>

Table 1: Comparison of various methods for image generation on LSUN Bedrooms, LSUN Churches, and FFHQ. All images are of resolution 256 × 256. BLD\* denotes our reproduction of BLD using its official codebase under the same machine environment and training configuration as BFM. The reproduced $\mathrm { B L D ^ { * } }$ and BFM results are evaluated from checkpoints trained for 800K iterations. Within each method category, the best and second-best results in each column are highlighted in bold and underline, respectively.

Beyond the quantitative improvements presented in Table 1, we also provide a qualitative assessment to showcase BFM’s ability to generate high-fidelity images across diverse datasets. Representative samples are shown in Fig. 3. Additionally, Appendix B provides extended visualizations demonstrating how increasing the sampling temperature shifts the outputs from stable, high-fidelity structures to greater visual diversity.

![](images/864b4392b8f3f3abd375a6dbb5b51978da55ac631934b8f1fb7101e7ee18a89f.jpg)  
LSUN Bedrooms

![](images/329e38ba36b100ce1bb47ac2ebb9369d8ec2cc2130e9bae192e43f1ca4916202.jpg)

![](images/c31955108dfccac8a4706ce2e876d2fd987404b1f1af58ebe701a6b8366581b8.jpg)  
FFHQ  
Figure 3: Samples from BFM on LSUN-Bedrooms, LSUN-Churches and FFHQ datasets. All samples resolution are 256x256.

## 4.4 Performance on Binarized MNIST

![](images/d6e36045ba7f9e6118deb660d86e55677e06bef6a40911daa0b30cb5a0a71e3d.jpg)

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Base method</td><td rowspan=1 colspan=1>FID</td></tr><tr><td rowspan=1 colspan=1>SFM [6]</td><td rowspan=1 colspan=1>Flow Matching</td><td rowspan=1 colspan=1>4.62</td></tr><tr><td rowspan=1 colspan=1>α-Flow [5]</td><td rowspan=1 colspan=1>Flow Matching</td><td rowspan=1 colspan=1>5.02</td></tr><tr><td rowspan=1 colspan=1>DMPM [23]</td><td rowspan=1 colspan=1>Score-based Diffusion</td><td rowspan=1 colspan=1>2.89</td></tr><tr><td rowspan=1 colspan=1>CoVAE [26]</td><td rowspan=1 colspan=1>VAE</td><td rowspan=1 colspan=1>0.58</td></tr><tr><td rowspan=1 colspan=1>BFM (ours)</td><td rowspan=1 colspan=1>Flow Matching &amp; Diffusion</td><td rowspan=1 colspan=1>0.42</td></tr></table>

Figure 4: Left: Binarized MNIST samples generated by BFM. Right: Comparison of FID scores. The best and second-best results are highlighted in bold and underline, respectively.

To validate the generative performance of BFM on high-dimensional binary manifolds, we conduct experiments on the binarized MNIST [8] dataset. Following standard practices, we apply a thresholding scheme where pixels greater than 0 are set to 1, and others are set to 0. We evaluate the performance of BFM and report the FID scores in Figure 4, comparing it with other state-of-the-art methods.

As shown, BFM achieves the best FID score of 0.42 among the evaluated models. The FID metric used to evaluate our generated binarized MNIST samples strictly follows the codebase provided in the SFM [6] repository. We visualize the generated high-quality samples on the left side of Figure 4.

## 4.5 Ablation Study on Bernoulli Probability Path Scheduling Strategies

Fig. 5 visualizes how, under different probability paths, the probabilities associated with 1 (blue curves) and 0 (red curves) are progressively driven towards 0.5 along the diffusion trajectory. To systematically assess the impact of these schedulers on model performance, we select 3 different probability path schedulers and conduct experiments on the LSUN Churches 256×256 dataset using the official BLD codebase. We keep the network architecture, learning rate, loss function, and all optimization hyperparameters exactly the same, and train with a batch size of 96. The only difference lies in the algorithm at the training and sampling stages.

![](images/f82e5053e77aab3a2b965461489cd513be5ee9dcc5b691ce6c8198ddbe77b37f.jpg)  
(a) Linear

![](images/5eae48f29cbd6b9c2a9690312015dc916a1165670ae83bbd88f67d37c4fefc9f.jpg)  
(b) Half cos

![](images/52c4345064720b3a4165804b5a7d2f0ad4cf5ba7aed1691e4f210acb411ad7d2.jpg)  
(c) Reverse half cos  
Figure 5: Bernoulli probability path schedulers.

We summarize the ablation results in Table 2. Under   
all three probability paths (linear, half-cosine, and re  
verse half-cosine), BFM consistently achieves lower   
FID than BLD. Among them, the linear scheduler   
gives the best overall performance for BFM (9.22).   
We also note that the FID of BFM varies slightly   
more across different schedulers than that of BLD, al  
though the overall variation remains small. This indi  
cates that while the specific choice of scheduling func  
tion has a measurable impact on the final generation   
quality, BFM maintains stable performance across   
different probability paths. Consequently, given its   
superior empirical performance, we adopt the linea r schedule as the default configuration for all subsequent BFM experiments. subsequent BFM experiments.

Table 2: FID comparison of different Bernoulli probability path schedulers for BFM and BLD on LSUN Churches 256×256, evaluated at 100K training iterations.
<table><tr><td rowspan="2">Scheduler</td><td colspan="2">FID↓</td></tr><tr><td>BFM</td><td>BLD</td></tr><tr><td>Linear</td><td>9.22</td><td>9.55</td></tr><tr><td>Half-cosine</td><td>9.41</td><td>9.53</td></tr><tr><td>Reverse half-cos</td><td>9.27</td><td>9.54</td></tr></table>

## 5 Discussion

We presented Bernoulli Flow Models (BFM), a self-consistent generative framework that defines a continuous global Bernoulli probability path between data and noise and derives analytical posterior transitions over arbitrary time intervals. This construction decouples inference from a fixed discrete grid: when the NFE changes, BFM re-evaluates the posterior on the new intervals rather than approximating skipped one-step transitions. Experiments show robust training-free low-NFE sampling while remaining competitive under standard inference. Our evaluation is currently limited to representative image-generation and scientific binary benchmarks, leaving broader domain-specific validation for future work. We also plan to combine BFM with distillation or consistency-style training for very-few-step or one-step generation.

## References

[1] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

[2] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.

[3] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. arXiv preprint, 2024.

[4] Ting Chen, Ruixiang Zhang, and Geoffrey E. Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. ArXiv, abs/2208.04202, 2022.

[5] Chaoran Cheng, Jiahan Li, Jiajun Fan, and Ge Liu. α-Flow: A Unified Framework for Continuous-State Discrete Flow Matching Models. arXiv preprint arXiv:2504.10283, 2025.

[6] Chaoran Cheng, Jiahan Li, Jian Peng, and Ge Liu. Categorical flow matching on statistical manifolds. Advances in Neural Information Processing Systems, 37:54787–54819, 2024.

[7] Oscar Davis, Samuel Kessler, Mircea Petrache, Ismail Ceylan, Michael Bronstein, and Joey Bose. Fisher flow matching for generative modeling over discrete data. Advances in Neural Information Processing Systems, 37:139054–139084, 2024.

[8] Li Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. IEEE signal processing magazine, 29(6):141–142, 2012.

[9] Zheng Ding, Mengqi Zhang, Jiajun Wu, and Zhuowen Tu. Patched denoising diffusion models for high-resolution image synthesis. In The twelfth international conference on learning representations, 2023.

[10] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. Advances in Neural Information Processing Systems, 37:133345–133385, 2024.

[11] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. Advances in Neural Information Processing Systems, 37:133345–133385, 2024.

[12] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10696–10706, 2022.

[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[14] Jiadong Hong, Lei Liu, Xinyu Bian, Wenjie Wang, and Zhaoyang Zhang. Binary flow matching: Prediction-loss space alignment for robust learning. arXiv preprint arXiv:2602.10420, 2026.

[15] Ernst Ising. Beitrag zur theorie des ferromagnetismus. Zeitschriftfür Physik, 31(1):253–258, 1925.

[16] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.

[17] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020.

[18] Volodymyr Kinakh and Slavi Voloshynovskiy. Binary diffusion probabilistic model. arXiv preprint, 2025. Under review.

[19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[20] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

[21] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. arXiv preprint arXiv:2206.00927, 2022.

[22] Bac Nguyen, Chieh-Hsin Lai, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Stefano Ermon, and Yuki Mitsufuji. Improving vector-quantized image modeling with latent consistencymatching diffusion. arXiv preprint arXiv:2410.14758, 2024.

[23] Le-Tuyet-Nhi Pham, Dario Shariatian, Antonio Ocello, Giovanni Conforti, and Alain Durmus. Discrete markov probabilistic models. arXiv preprint arXiv:2502.07939, 2025.

[24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[25] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

[26] Gianluigi Silvestri and Luca Ambrogioni. Covae: Consistency training of variational autoencoders. arXiv preprint arXiv:2507.09103, 2025.

[27] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.

[28] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[29] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

[30] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[31] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[32] Ze Wang, Jiang Wang, Zicheng Liu, and Qiang Qiu. Binary latent diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22576–22585, 2023.

[33] Ulli Wolff. Collective monte carlo updating for spin systems. Physical Review Letters, 62(4):361, 1989.

[34] Julia Wolleb, Florentin Bieder, Paul Friedrich, Peter Zhang, Alicia Durrer, and Philippe C Cattin. Binary noise for binary tasks: Masked bernoulli diffusion for unsupervised anomaly detection. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 135–145. Springer, 2024.

[35] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.

## A Proofs of Bernoulli Probability Flow Dynamics

In this section, we provide the detailed and rigorous proofs for the Bernoulli Probability Flow Dynamics introduced in Section 3.

## A.1 Mathematical Preliminaries

We first formalize the core algebraic operator used to simplify the binary probability space.

Definition 1 (Symmetric Binary Combination). We define the symmetric combination function $g : [ 0 , 1 ] \times [ 0 , 1 ] ^ { \cdot }  [ 0 , 1 ]$ as:

$$
g ( a , b ) = a ( 1 - b ) + b ( 1 - a ) .\tag{10}
$$

## A.2 Marginal Distribution Dynamics

We begin by establishing the marginal distribution from the data $X _ { t _ { 0 } }$ to an arbitrary state $X _ { t _ { \tau } }$

Proposition 1 (Marginal Probability). Let $X _ { t _ { 0 } } \in \{ 0 , 1 \}$ be a data sample and $X _ { 1 } \sim \mathit { B } ( 0 . 5 )$ be a uniform prior. For a sequence of time parameters $\dot { t } _ { \tau } \in [ 1 , 0 ]$ , let $X _ { t _ { \tau } }$ be defined conditionally as $X _ { t _ { \tau } } \sim \mathcal { B } \left( t _ { \tau } X _ { t _ { 0 } } + ( 1 - t _ { \tau } ) X _ { 1 } \right)$ . The exact marginal distribution is given by:

$$
q ( X _ { t _ { \tau } } \mid X _ { t _ { 0 } } ) = B \left( X _ { t _ { \tau } } ; g \left( X _ { t _ { 0 } } , \frac { 1 - t _ { \tau } } { 2 } \right) \right) .\tag{11}
$$

Proof. We aim to derive the conditional distribution $q ( X _ { t _ { \tau } } \mid X _ { t _ { 0 } } )$ by marginalizing over the uniform prior $X _ { 1 } ~ \sim ~ { \cal B } ( 0 . 5 )$ . By definition, the Bernoulli parameter for the intermediate state is $p =$ $t _ { \tau } X _ { t _ { 0 } } + ( 1 - t _ { \tau } ) X _ { 1 }$ . Using the linearity of expectation, the marginal probability of observing state 1 is computed directly as:

$$
\begin{array} { r l } { \mathbb { P } ( X _ { t _ { \tau } } = 1 \mid X _ { t _ { 0 } } ) = \mathbb { E } _ { X _ { 1 } } \left[ t _ { \tau } X _ { t _ { 0 } } + ( 1 - t _ { \tau } ) X _ { 1 } \right] } & { } \\ { = t _ { \tau } X _ { t _ { 0 } } + ( 1 - t _ { \tau } ) \cdot \mathbb { E } [ X _ { 1 } ] } & { } \\ { = t _ { \tau } X _ { t _ { 0 } } + 0 . 5 ( 1 - t _ { \tau } ) . } \end{array}
$$

Notice that this compact expression naturally evaluates to $0 . 5 ( 1 - t _ { \tau } )$ when $X _ { t _ { 0 } } = 0$ , and to $0 . 5 ( 1 { + } t _ { \tau } )$ when $X _ { t _ { 0 } } = 1$

To unify this parameter into a symmetric form, we algebraically rewrite the expression as:

$$
\alpha _ { t _ { \tau } } : = \mathbb { P } ( X _ { t _ { \tau } } = 1 \mid X _ { t _ { 0 } } ) = { \frac { 1 + ( 2 X _ { t _ { 0 } } - 1 ) t _ { \tau } } { 2 } } .\tag{12}
$$

It is straightforward to algebraically verify that $\begin{array} { r } { g \left( X _ { t _ { 0 } } , \frac { 1 - t _ { \tau } } { 2 } \right) = X _ { t _ { 0 } } \left( 1 - \frac { 1 - t _ { \tau } } { 2 } \right) + \frac { 1 - t _ { \tau } } { 2 } ( 1 - X _ { t _ { 0 } } ) } \end{array}$ which simplifies exactly to $\frac { 1 + ( 2 X _ { t _ { 0 } } - 1 ) t _ { \tau } } { 2 }$ . Therefore, the marginal probability is exactly parameterized by $\begin{array} { r } { \alpha _ { t _ { \tau } } = g \left( X _ { t _ { 0 } } , \frac { 1 - t _ { \tau } } { 2 } \right) } \end{array}$ □

## A.3 One-Step Forward Transition Dynamics

We now prove the transition probability between two sequential states $X _ { t _ { \tau - 1 } }$ and $X _ { t _ { \tau } }$

Proposition 2 (Forward Transition Probability). Assuming a Markovian transition process from $X _ { t _ { \tau - 1 } } t o X _ { t } .$ with an error rate $\gamma _ { t _ { \tau } }$ , the transition probability is uniquely specified by:

$$
q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - 1 } } ) = \mathcal { B } \left( X _ { t _ { \tau } } ; g ( X _ { t _ { \tau - 1 } } , \gamma _ { t _ { \tau } } ) \right) , \quad w h e r e \quad \gamma _ { t _ { \tau } } = \frac { 0 . 5 ( t _ { \tau - 1 } - t _ { \tau } ) } { t _ { \tau - 1 } } .\tag{13}
$$

Proof. Let $\alpha _ { t _ { \tau } }$ and $\alpha _ { t _ { \tau - 1 } }$ be the marginal probabilities of state 1 at their respective time steps, defined via Eq. 12. By the law of total probability, the marginal probability $\alpha _ { t }$ must aggregate all paths from the previous state:

$$
\alpha _ { t _ { \tau } } = \sum _ { x \in \{ 0 , 1 \} } q ( X _ { t _ { \tau } } = 1 \mid X _ { t _ { \tau - 1 } } = x ) \cdot q ( X _ { t _ { \tau - 1 } } = x \mid X _ { t _ { 0 } } ) .\tag{14}
$$

Substituting the parameterized transition probabilities yields:

$$
\alpha _ { t _ { \tau } } = \alpha _ { t _ { \tau - 1 } } ( 1 - \gamma _ { t _ { \tau } } ) + ( 1 - \alpha _ { t _ { \tau - 1 } } ) \gamma _ { t _ { \tau } } .\tag{15}
$$

Rearranging the terms and solving for the flip probability $\gamma _ { t _ { \tau } }$ :

$$
\alpha _ { t _ { \tau } } = \alpha _ { t _ { \tau - 1 } } - 2 \gamma _ { t _ { \tau } } \alpha _ { t _ { \tau - 1 } } + \gamma _ { t _ { \tau } } \implies \gamma _ { t _ { \tau } } = \frac { \alpha _ { t _ { \tau - 1 } } - \alpha _ { t _ { \tau } } } { 2 \alpha _ { t _ { \tau - 1 } } - 1 } .\tag{16}
$$

Substituting $\begin{array} { r } { \alpha _ { t _ { \tau } } = \frac { 1 + ( 2 X _ { t _ { 0 } } - 1 ) t _ { \tau } } { 2 } } \end{array}$ and $\begin{array} { r } { \alpha _ { t _ { \tau - 1 } } = \frac { 1 + ( 2 X _ { t _ { 0 } } - 1 ) t _ { \tau - 1 } } { 2 } } \end{array}$ into the equation exactly cancels out the data-dependent term $( \overline { { 2 } } X _ { t _ { 0 } } - 1 )$ , resulting in:

$$
\gamma _ { t _ { \tau } } = \frac { \frac { 1 } { 2 } ( 2 X _ { t _ { 0 } } - 1 ) ( t _ { \tau - 1 } - t _ { \tau } ) } { ( 2 X _ { t _ { 0 } } - 1 ) t _ { \tau - 1 } } = \frac { 0 . 5 ( t _ { \tau - 1 } - t _ { \tau } ) } { t _ { \tau - 1 } } .\tag{17}
$$

This confirms the exact functional form of the forward transition.

## A.4 Analytical One-Step Posterior Derivation

Proposition 3 (One-Step Closed-Form Posterior). The ground-truth posterior distribution of $X _ { t _ { \tau - 1 } }$ conditioned on $X _ { t _ { \tau } }$ and the source data $X _ { t _ { 0 } }$ admits the exact closed-form solution:

$$
q ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = \mathcal { B } \left( X _ { t _ { \tau - 1 } } ; \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) \alpha _ { t _ { \tau - 1 } } } { g ( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) , 1 - \alpha _ { t _ { \tau - 1 } } ) } \right) .\tag{18}
$$

Proof. Our objective is to find the analytical Bernoulli parameter for the reverse transition, which requires calculating the probability of $\dot { X } _ { t _ { \tau - 1 } } = 1$ using Bayes’ theorem:

$$
q ( X _ { t _ { \tau - 1 } } = 1 \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = { \frac { q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - 1 } } = 1 ) \cdot q ( X _ { t _ { \tau - 1 } } = 1 \mid X _ { t _ { 0 } } ) } { \sum _ { x \in \{ 0 , 1 \} } q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - 1 } } = x ) \cdot q ( X _ { t _ { \tau - 1 } } = x \mid X _ { t _ { 0 } } ) } } .\tag{19}
$$

Based on the dynamics established in Propositions 1 and 2, we define the constituent transition probabilities:

$$
\begin{array} { r l r l } & { q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - 1 } } = 1 ) = g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) , } & & { q ( X _ { t _ { \tau - 1 } } = 1 \mid X _ { t _ { 0 } } ) = \alpha _ { t _ { \tau - 1 } } , } \\ & { q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - 1 } } = 0 ) = 1 - g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) , } & & { q ( X _ { t _ { \tau - 1 } } = 0 \mid X _ { t _ { 0 } } ) = 1 - \alpha _ { t _ { \tau - 1 } } . } \end{array}\tag{20}
$$

Substituting these precise terms into the fractional Bayes’ formulation in Eq. 19, we obtain:

$$
q ( X _ { t _ { \tau - 1 } } = 1 \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = { \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) \alpha _ { t _ { \tau - 1 } } } { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) \alpha _ { t _ { \tau - 1 } } + ( 1 - g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) ) \left( 1 - \alpha _ { t _ { \tau - 1 } } \right) } } .\tag{21}
$$

By recognizing the symmetric structure of the denominator, we apply the definition $g ( a , b ) =$ $a ( 1 - b ) + b ( 1 - a )$ , which folds the denominator perfectly into $g \left( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) , 1 - \alpha _ { t _ { \tau - 1 } } \right)$ . Thus, the final expression simplifies to:

$$
q ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = { \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) \alpha _ { t _ { \tau - 1 } } } { g ( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } } ) , 1 - \alpha _ { t _ { \tau - 1 } } ) } } ,\tag{22}
$$

concluding the derivation of the one-step posterior.

## A.5 Exact Cross-Step Posterior (Proof of Theorem 1)

Finally, we extend the one-step derivation to any arbitrary interval to prove the fast-sampling mechanism introduced in the main text.

Theorem 1 (Exact Cross-Step Posterior, restated). For any continuous time steps $t _ { \tau - k }$ and $t _ { \tau }$ where $k > 1$ , the exact cross-step posterior ofthe Bernoulli Flow Model admits a perfect closed-form solution:

$$
q ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = \mathcal { B } \left( X _ { t _ { \tau - k } } ; \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) \alpha _ { t _ { \tau - k } } } { g ( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) , 1 - \alpha _ { t _ { \tau - k } } ) } \right) .\tag{23}
$$

Proof. Because our parameters α and $\gamma$ are defined continuously across the interpolation path, the algebraic relationships proven in Propositions 1 and 2 hold for any arbitrary temporal stride. To jump from $t _ { \tau }$ directly back to $t _ { \tau - k }$ , we generalize the transition error rate over this specific interval as $\begin{array} { r } { \gamma _ { t _ { \tau } | t _ { \tau - k } } = \frac { 0 . 5 ( t _ { \tau - k } - t _ { \tau } ) } { t _ { \tau - k } } } \end{array}$

Using Bayes’ theorem, the probability of $X _ { t _ { \tau - k } } = 1$ is calculated exactly as:

$$
q ( X _ { t _ { \tau - k } } = 1 \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = { \frac { q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - k } } = 1 ) \cdot q ( X _ { t _ { \tau - k } } = 1 \mid X _ { t _ { 0 } } ) } { \sum _ { x \in \{ 0 , 1 \} } q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - k } } = x ) \cdot q ( X _ { t _ { \tau - k } } = x \mid X _ { t _ { 0 } } ) } } .\tag{24}
$$

Substituting the generalized constituent parameters for the k-step stride:

$$
\begin{array} { r l r l } & { q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - k } } = 1 ) = g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) , } & & { q ( X _ { t _ { \tau - k } } = 1 \mid X _ { t _ { 0 } } ) = \alpha _ { t _ { \tau - k } } , } \\ & { q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - k } } = 0 ) = 1 - g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) , } & & { q ( X _ { t _ { \tau - k } } = 0 \mid X _ { t _ { 0 } } ) = 1 - \alpha _ { t _ { \tau - k } } . } \end{array}\tag{25}
$$

Inserting these into Eq. 24 yields the expanded fraction:

$$
q ( X _ { t _ { \tau - k } } = 1 \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) \alpha _ { t _ { \tau - k } } } { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) \alpha _ { t _ { \tau - k } } + \left( 1 - g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) \right) ( 1 - \alpha _ { t _ { \tau - k } } ) } .\tag{26}
$$

Applying the symmetric operator definition $g ( a , b ) = a ( 1 - b ) + b ( 1 - a )$ to the denominator precisely folds it into $g \left( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) , 1 - \alpha _ { t _ { \tau - k } } \right)$ . Consequently, the analytical cross-step posterior is cleanly obtained:

$$
q ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) \alpha _ { t _ { \tau - k } } } { g ( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) , 1 - \alpha _ { t _ { \tau - k } } ) } ,\tag{27}
$$

which theoretically guarantees the model’s structural consistency under low-NFE inference. □

## A.6 Connecting the Analytical Posterior to the Practical Sampler

This subsection clarifies how the clean-data-conditioned posterior in Theorem 1 is related to the practical reverse transition used during generation. Specifically, we explain three points: why the posterior conditioned on $X _ { t _ { 0 } }$ is available as a ground-truth denoising target during training, why likelihood-based training requires the learned reverse transition to approximate this posterior, and why the resulting reverse transition can be parameterized by predicting $X _ { t _ { 0 } }$ rather than directly predicting the complete posterior probability.

Forward and reverse Bernoulli processes. Let

$$
1 = t _ { 0 } > t _ { 1 } > \cdot \cdot \cdot > t _ { T } = 0\tag{28}
$$

denote a finite time grid. Here, $X _ { t _ { 0 } }$ is a clean binary observation drawn from the data distribution $\pi _ { 0 }$ whereas $X _ { t _ { T } }$ follows the Bernoulli prior $\pi _ { 1 } = B ( 0 . 5 )$

The predefined forward process progressively transforms the clean data into Bernoulli noise:

$$
q ( X _ { t _ { 1 : T } } \mid X _ { t _ { 0 } } ) = \prod _ { \tau = 1 } ^ { T } q ( X _ { t _ { \tau } } \mid X _ { t _ { \tau - 1 } } ) .\tag{29}
$$

All transitions in this process are analytically specified and contain no learnable parameters.

The generative model operates in the opposite direction:

$$
p _ { \theta } ( X _ { t _ { 0 : T } } ) = p ( X _ { t _ { T } } ) \prod _ { \tau = 1 } ^ { T } p _ { \theta } ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } ) .\tag{30}
$$

Generation begins from $X _ { t _ { T } } \sim B ( 0 . 5 )$ and repeatedly samples from the learned reverse transitions until a clean sample $X _ { t _ { 0 } }$ is obtained. The learning problem is therefore to construct $p _ { \theta } ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } )$ using only the current noisy state $X _ { t _ { \tau } }$

Why the oracle posterior appears during training. Ideally, the reverse sampler should reproduce the true reverse transition

$$
q ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } ) .\tag{31}
$$

This distribution is generally intractable because evaluating it requires marginalizing over the unknown data distribution. During training, however, the clean sample ${ \bar { X } } _ { t _ { 0 } }$ is observed from the dataset, and $X _ { t _ { \tau } }$ is generated by applying the predefined forward process to this known sample. We can therefore evaluate the tractable conditional posterior

$$
q ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) .\tag{32}
$$

We refer to this distribution as the oracle posterior because it has access to the clean state toward which $X _ { t }$ should be denoised. The conditioning variable $X _ { t _ { 0 } }$ is available only when constructing the training target; it is not assumed to be available during generation.

The role of this oracle posterior follows directly from maximum-likelihood training. The objective of the generative model is to maximize the likelihood assigned to the observed clean data, $\log p _ { \theta } ( X _ { t _ { 0 } } )$ Marginalizing over the intermediate binary states gives

$$
\begin{array} { l } { \log p _ { \theta } ( X _ { t _ { 0 } } ) = \log \displaystyle \sum _ { X _ { t _ { 1 } } , \ldots , X _ { t _ { T } } } p _ { \theta } \big ( X _ { t _ { 0 } } , X _ { t _ { 1 } } , \ldots , X _ { t _ { T } } \big ) } \\ { = \log \displaystyle \sum _ { X _ { t _ { 1 } } , \ldots , X _ { t _ { T } } } p _ { \theta } \big ( X _ { t _ { 0 : T } } \big ) \frac { q \big ( X _ { t _ { 1 : T } } \big \vert \ X _ { t _ { 0 } } \big ) } { q \big ( X _ { t _ { 1 : T } } \big \vert \ X _ { t _ { 0 } } \big ) } } \\ { = \log \displaystyle \sum _ { X _ { t _ { 1 } } , \ldots , X _ { t _ { T } } } q \big ( X _ { t _ { 1 : T } } \big \vert \ X _ { t _ { 0 } } \big ) \frac { p _ { \theta } \big ( X _ { t _ { 0 : T } } \big ) } { q \big ( X _ { t _ { 1 : T } } \big \vert \ X _ { t _ { 0 } } \big ) } } \\ { = \log \mathbb { E } _ { q ( X _ { t _ { 1 : T } } \mid X _ { t _ { 0 } } ) } \left[ \frac { p _ { \theta } \big ( X _ { t _ { 0 : T } } \big ) } { q ( X _ { t _ { 1 : T } } \big \vert \ X _ { t _ { 0 } } \big ) } \right] . } \end{array}\tag{33}
$$

Applying Jensen’s inequality produces a tractable lower bound:

$$
\log p _ { \theta } ( X _ { t _ { 0 } } ) \geq \mathbb { E } _ { q } \left[ \log \frac { p _ { \theta } ( X _ { t _ { 0 : T } } ) } { q ( X _ { t _ { 1 : T } } \mid X _ { t _ { 0 } } ) } \right] .\tag{34}
$$

Substituting the forward factorization in Eq. (29) and the reverse factorization in Eq. (30), and rearranging the forward transitions using Bayes’ rule, decomposes this lower bound as

$$
\begin{array} { r l } { \log p _ { \theta } ( X _ { t _ { 0 } } ) \geq \mathbb { E } _ { q } \left[ \log p _ { \theta } ( X _ { t _ { 0 } } \mid X _ { t _ { 1 } } ) \right] } & { } \\ { \displaystyle - D _ { \mathrm { K L } } \left( q ( X _ { t _ { T } } \mid X _ { t _ { 0 } } ) \| p ( X _ { t _ { T } } ) \right) } & { } \\ { \displaystyle - \sum _ { \tau = 2 } ^ { T } \mathbb { E } _ { q ( X _ { t _ { \tau } } \mid X _ { t _ { 0 } } ) } \left[ D _ { \mathrm { K L } } \left( q ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) \| p _ { \theta } ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } ) \right) \right] . } & { } \end{array}\tag{35}
$$

Equation (35) makes the learning objective explicit. The first term encourages reconstruction of the clean sample from the least corrupted state. The second term matches the endpoint of the forward process to the prescribed Bernoulli prior and contains no learnable reverse-transition parameters. Most importantly, maximizing the lower bound requires minimizing the denoising-matching terms

$$
D _ { \mathrm { K L } } \left( q ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) \parallel p _ { \theta } ( X _ { t _ { \tau - 1 } } \mid X _ { t _ { \tau } } ) \right) .\tag{36}
$$

Thus, the oracle posterior is not an additional assumption made at inference time. It is the tractable ground-truth reverse transition that the practical sampler is trained to approximate. By making these per-step reverse transitions accurate, the learned process can iteratively transform samples from the Bernoulli prior back toward the data distribution.

Why predicting $X _ { t _ { 0 } }$ is sufficient. Equation (36) specifies the learning target: the practical reverse transition $p _ { \theta } ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } )$ should approximate the ground-truth posterior $\mathsf { \bar { q } } ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } )$ . It does not, however, specify how $p _ { \theta }$ should be parameterized.

One possible way to parameterize the practical reverse transition is to use a neural network to directly predict the complete Bernoulli probability of $p _ { \theta } ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } )$ and train it to match the Bernoulli parameter of the ground-truth posterior $q ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } )$ . In other words, the network could directly regress the full oracle posterior probability for each noisy state and time interval. However, as noted in BLD [32], directly modeling such reverse-transition probabilities can be challenging, because the model must accurately regress sophisticated interpolations that vary jointly with the current noisy state and the noise level.

In BFM, Theorem 1 provides the analytical form of this posterior. For an arbitrary reverse interval from $t _ { \tau }$ to $t _ { \tau - k }$ , the oracle posterior is

$$
q ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } , X _ { t _ { 0 } } ) = \mathcal { B } \left( X _ { t _ { \tau - k } } ; \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) \alpha _ { t _ { \tau - k } } } { g \left( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } | t _ { \tau - k } } ) , 1 - \alpha _ { t _ { \tau - k } } \right) } \right) ,\tag{37}
$$

where

$$
\alpha _ { t _ { \tau - k } } = g \left( X _ { t _ { 0 } } , \frac { 1 - t _ { \tau - k } } { 2 } \right) , \qquad \gamma _ { t _ { \tau } | t _ { \tau - k } } = \frac { 0 . 5 ( t _ { \tau - k } - t _ { \tau } ) } { t _ { \tau - k } } .\tag{38}
$$

For a given reverse update, $X _ { t _ { \tau } }$ and the inference times are known, and $\gamma _ { t \tau | t _ { \tau - k } }$ is analytically determined. Thus, the only unknown quantity in Eq. (37) is $X _ { t _ { 0 } } .$ We use the neural network $f _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } )$ to predict $X _ { t _ { 0 } }$ . Hence, the posterior p<sub>θ</sub> parameterized by $f _ { \theta }$ can be written as

$$
\begin{array} { r l } & { p _ { \theta } ( X _ { t _ { \tau - k } } \mid X _ { t _ { \tau } } ) } \\ & { = \mathcal { B } \left( X _ { t _ { \tau - k } } ; \frac { g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } \mid t _ { \tau - k } } ) g \left( f _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } ) , \frac { 1 - t _ { \tau - k } } { 2 } \right) } { g \left( g ( X _ { t _ { \tau } } , \gamma _ { t _ { \tau } \mid t _ { \tau - k } } ) , 1 - g \left( f _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } ) , \frac { 1 - t _ { \tau - k } } { 2 } \right) \right) } \right) . } \end{array}\tag{39}
$$

Therefore, $X _ { t _ { 0 } }$ -prediction provides a structured parameterization of posterior matching. In the ideal case where $f _ { \theta } ( X _ { t _ { \tau } } , t _ { \tau } ) = X _ { t _ { 0 } }$ , Eq. (39) exactly recovers the oracle posterior in Eq. (37). This reparameterization through $X _ { t _ { \Omega } }$ -prediction corresponds to the standard x<sub>0</sub>-prediction strategy widely used in DDPMs and SDE diffusion models [13, 30].

![](images/74229abbc40b22ea10353bd29daa31cab0d7783657612006c591c90715bf3efe.jpg)  
Figure 6: Unconditional 256 × 256 samples on LSUN Bedrooms. Sampling temperatures κ are linearly interpolated from 0.5 to 1.0 (left to right).

![](images/b4f5f3b5d38e88b447bf96a61e3cfbf84867479a6a7ee0f1ba3c5372ee6cafde.jpg)  
Figure 7: Unconditional 256 × 256 samples on LSUN Churches. Sampling temperatures κ are linearly interpolated from 0.5 to 1.0 (left to right).

![](images/a021ab23f37eee09c068a6d4e30e7dfb91f4f463a113b0b8a21a7b4e7b258531.jpg)  
Figure 8: Unconditional 256 × 256 samples on FFHQ. Sampling temperatures κ are linearly interpolated from 0.5 to 1.0 (left to right).

## C Beyond Image Synthesis: Ising Model Generation

Finally, we evaluate BFM on the Ising model, a canonical probabilistic model in statistical physics defined over interacting spin variables s $\in \{ + 1 , - 1 \} ^ { D }$ [15]. This experiment demonstrates that BFM is not restricted to image generation, but also applies to general binary scientific data.

![](images/6c58501b463188483e1a99b05be1575404d6a5d217c61e85c24137108484e057.jpg)  
Figure 9: Comparison of 20 × 20 Ising lattice samples generated by BFM and ground-truth samples at temperatures 2.0 and 2.5.

To obtain reference Ising samples at different temperatures, we use the Wolff cluster algorithm [33]. We consider a ${ \mathrm { . 2 0 } } \times { \mathrm { 2 0 } } { \mathrm { 2 D } }$ lattice Ising model with periodic boundary conditions, where each spin interacts with its four nearest neighbors. The Markov chain is run for 1000 steps to reach thermal equilibrium. We collect 500 training samples and 100 validation samples at each temperature from 1.50 to 3.10. We then train a conditional BFM for 100K iterations with batch size 512. During inference, we use 64 sampling steps to generate Ising samples from pure noise.

Fig. 9 shows representative comparisons between BFM-generated samples and Wolff-cluster reference samples at temperatures 2.0 and 2.5. BFM generates high-quality Ising configurations that are visually consistent with the reference samples.

![](images/32e63f71f73e8d50caf03904ee93628d0284d51a7ff97d0e5c690a12012b2c9c.jpg)  
(a) BFM Ising Samples

![](images/3774fa9f73804aa2b37e403706d08d913e0cec50ce05204c0b83bdcd1aaa277c.jpg)  
(b) Wolff Cluster Ising Samples

![](images/3d3c94eb707e08273013ac3b65683f3b7fa5946e678a2f755a9299857a0814ef.jpg)  
(c) Magnetization Curve  
Figure 10: Comprehensive evaluation of Ising model generation across the continuous temperature spectrum (1.50 to 3.10). (a) and (b) show the visual evolution of spin lattices generated by BFM and the ground-truth Wolff cluster [33] algorithm, respectively. (c) Quantitative comparison of the absolute magnetization $M ( T )$ , demonstrating that BFM accurately captures the thermodynamic properties.

To further investigate our model’s ability to simulate the macroscopic thermodynamic behavior of the Ising system, we extend our evaluation across the entire temperature spectrum in Fig. 10. In statistical physics, visual inspection is often insufficient; a generative model must faithfully capture the underlying Boltzmann distribution. To rigorously substantiate our claim, we calculate the absolute magnetization $M ( T )$ of the generated samples as a quantitative physical observable.

Visually, the BFM-generated samples (Fig. 10a) smoothly capture the structural evolution of the spin lattices from highly ordered to completely disordered states, strictly mirroring the Wolff cluster ground truth (Fig. 10b). More importantly, the quantitative magnetization curve of BFM (Fig. 10c) tightly tracks the ground truth simulations. This quantitative alignment proves that BFM learns the complex physical interaction rules of the system rather than merely memorizing visual training patterns.

## D Discussion on Cross-step Sampling Consistency of BFM

In this section, we discuss the reason why BFM exhibits better sampling quality than BLD when using different NFE from the same trained checkpoint. This property is particularly important in practical applications, as it allows for flexible adjustment of sampling speed and quality trade-offs without the need for retraining the model. We follow the notation used in BLD to explain this issue. In the sampling scenario of Fig. 2, when we sample with K NFE from a model trained with N steps, we need to perform cross-step sampling with stride $m = N / K$ . According to Eq. (9) in BLD, the posterior transition can be written as

$$
\begin{array} { r } { q ( \mathbf { z } ^ { t - m } | \mathbf { z } ^ { t } , \mathbf { z } ^ { 0 } ) = \frac { q ( \mathbf { z } ^ { t } | \mathbf { z } ^ { t - m } , \mathbf { z } ^ { 0 } ) q ( \mathbf { z } ^ { t - m } | \mathbf { z } ^ { 0 } ) } { q ( \mathbf { z } ^ { t } | \mathbf { z } ^ { 0 } ) } . } \end{array}\tag{40}
$$

In the official BLD code implementation, the evidence transition term is implemented as

$$
\begin{array} { r } { q ( \mathbf { z } ^ { t } | \mathbf { z } ^ { t - m } , \mathbf { z } ^ { 0 } ) = \mathcal { B } ( \mathbf { z } ^ { t } ; \mathbf { z } ^ { t - m } ( 1 - \beta ^ { t - m } ) + 0 . 5 \beta ^ { t - m } ) , } \end{array}\tag{41}
$$

which effectively uses only the last-step noise parameter $\beta ^ { t - m }$ for this cross-step update; however, the corresponding exact evidence transition should be

$$
q ( \mathbf { z } ^ { t } \mid \mathbf { z } ^ { t - m } ) = \mathcal { B } \left( \mathbf { z } ^ { t } ; \left( \prod _ { j = t - m + 1 } ^ { t } ( 1 - \beta ^ { j } ) \right) \mathbf { z } ^ { t - m } + 0 . 5 \left( 1 - \prod _ { j = t - m + 1 } ^ { t } ( 1 - \beta ^ { j } ) \right) \right)\tag{42}
$$

In contrast, in the framework of BFM, the cross-step posterior sampling is naturally supported by our derived posterior transition in (3) and (5), which can be directly applied to any arbitrary cross-step posterior transition with slight modification. The only modification is to reparameterize the linear interpolation of $t _ { \tau }$ on [1, 0.5] from the original N training steps to a new number of NFE K at sampling time. We refer to this property as the self-consistency of BFM under cross-step sampling.

Here we explain why BFM has this self-consistency property. The evidence transition $q ( X _ { t _ { \tau } } | X _ { t _ { \tau - 1 } } , \bar { X } _ { t _ { 0 } } )$ from $X _ { t _ { \tau - 1 } } \mathrm { ~ t o ~ } X _ { t } .$ depends on $\gamma _ { t _ { \tau } }$ , we use the marginal distribution from $X _ { t _ { 0 } }$ to $X _ { t _ { \tau } }$ and $X _ { t _ { \tau } }$ to derive $\gamma _ { t . }$ and find it only depends on $t _ { \tau }$ and $t _ { \tau - 1 }$ as (17) shows. Therefore, when we perform cross-step sampling with different NFE, we can simply re-calculate $\gamma _ { t _ { \tau } }$ based on the new $t _ { \tau }$ and $t _ { \tau - 1 }$ without any approximation, which ensures the correctness of the posterior transition. This is the key mechanism of BFM’s self-consistency property in cross-step sampling. The self-consistency property makes BFM more flexible and robust in cross-step sampling scenarios.

## E Additional Implementation Details

In this section, we provide detailed network configurations and training hyperparameters for all tasks, as summarized in Table 3. For high-dimensional image synthesis (LSUN and FFHQ), the training process operates within a discrete latent space. Regarding these latent representations, we utilize the official pre-trained Binary Autoencoder (BAE) provided by BLD [32]. Rather than training a dedicated autoencoder for each specific dataset, we adopt their general-purpose C64 model, which was trained on over 600 million images from the LAION dataset. This robust autoencoder explicitly compresses a high-resolution 256 × 256 image into a compact $1 6 \times 1 6 \times 6 4$ binary tensor, providing a highly generalized discrete space for our Bernoulli Flow Models. The pre-trained checkpoints for this autoencoder (including both the encoder and decoder) are officially provided by the BLD authors and can be accessed publicly<sup>1</sup>. To ensure a fair comparison and robust performance in modeling these binary latents, our core generative network directly adopts the Transformer-based backbone proposed in BLD. While this standard configuration is seamlessly applied to the latent image representations, we introduce minor structural adaptations, specifically to the sequence length and positional embedding layers, to properly accommodate the unique spatial formats of Binarized MNIST and the 2D Ising model. Unlike the high-dimensional image tasks, Binarized MNIST [8] and the Ising model [15] do not rely on any encoder-decoder mappings; our models are trained directly on their native binary spaces. For the Ising model specifically, the spin configurations were synthetically simulated using the Wolff cluster [33] algorithm, yielding a custom dataset of 15000 training samples and 3000 testing samples. Across all datasets, we maintain a consistent optimization strategy using the Adam optimizer coupled with a linear learning rate warmup.

Table 3: Configuration and training details for BFM across different datasets.
<table><tr><td>Component</td><td>LSUN &amp; FFHQ</td><td>Binarized MNIST</td><td>Ising Model</td></tr><tr><td>Transformer layers</td><td>24</td><td>24</td><td>24</td></tr><tr><td>Attention heads</td><td>12</td><td>12</td><td>12</td></tr><tr><td>Embedding dimension</td><td>768</td><td>768</td><td>768</td></tr><tr><td>Sequence length (block size)</td><td>256</td><td>28</td><td>20</td></tr><tr><td>Training iterations</td><td>500K</td><td>500K</td><td>100K</td></tr><tr><td>Batch size</td><td>96</td><td>512</td><td>512</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning rate</td><td>2 × 10−4</td><td>2 × 10−4</td><td>2 × 10−4</td></tr><tr><td>Weight decay</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Warmup iterations</td><td>10,000</td><td>10,000</td><td>10,000</td></tr><tr><td>Hardware</td><td>2× RTX 4090</td><td>2× RTX 4090</td><td>2× RTX 4090</td></tr></table>

## E.1 Compute Resources and Inference Efficiency

To improve reproducibility and address the computational resource requirements of our experiments, we further report the inference-time compute cost for each task. All measurements are conducted with batch size 1 and 64 sampling steps, corresponding to the single-sample generation latency rather than throughput under large-batch sampling. For each benchmark, we first perform 5 warm-up runs to exclude CUDA initialization and caching overhead, and then average the wall-clock time over 20 repeated runs. We enforce CUDA synchronization before and after each measured run. For LSUN and FFHQ, we report a shared Image-256 setting because both tasks use the same 256 × 256 image resolution and the same $1 6 \times 1 6 \times 6 4$ binary latent representation produced by the pre-trained Binary Autoencoder (BAE). This benchmark includes both binary latent sampling and decoding through the BAE decoder. In contrast, Binarized MNIST and the Ising model are sampled directly in their native binary spaces without any encoder-decoder module.

Table 4: Single-sample inference resource usage. We report the average wall-clock latency and peak GPU memory for one generated sample using 64 sampling steps. The Image-256 setting is shared by LSUN and FFHQ since they use the same image resolution and binary latent format.
<table><tr><td>Task</td><td>Representation</td><td>Batch Size</td><td>Sampling Steps</td><td>Avg. Time / Sample (s)</td><td>Peak GPU Memory (GB)</td></tr><tr><td>LSUN / FFHQ Image-256</td><td>Binary latent + BAE decoder</td><td>1</td><td>64</td><td>0.3837</td><td>1.0743</td></tr><tr><td>Binarized MNIST</td><td>Native binary space</td><td>1</td><td>64</td><td>0.3507</td><td>0.6482</td></tr><tr><td>Ising Model</td><td>Native binary space</td><td>1</td><td>64</td><td>0.3531</td><td>0.6480</td></tr></table>

As shown in Table 4, all tasks can be sampled with less than 1.1 GB of peak GPU memory under the single-sample setting. The Image-256 benchmark has slightly higher latency and memory consumption because it operates on a higher-dimensional binary latent space and additionally invokes the BAE decoder to reconstruct RGB images. Binarized MNIST and Ising require less memory since they are generated directly in low-dimensional binary spaces. These measurements indicate that our 64-step sampler remains lightweight at inference time across both image-latent generation and native binary-domain generation.