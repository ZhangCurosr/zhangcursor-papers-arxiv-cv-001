# BEYOND BEST-OF-k: PRIOR-CONSISTENT DAPS FOR AMBIGUOUS NONLINEAR INVERSE PROBLEMS Merve Kocabas<sup>1,2,3\*</sup> Benjamin A. Burns<sup>4\*</sup> Andreas Geiger<sup>1,5,6,7</sup> Kent Gauen<sup>4</sup> Sara Fridovich-Keil<sup>4</sup>

<sup>1</sup>University of Tubingen¨ <sup>2</sup>Zuse School ELIZA

<sup>3</sup>Max Planck Institute for Intelligent Systems <sup>4</sup>Georgia Institute of Technology <sup>5</sup>Tubingen AI Center¨ <sup>6</sup>ELLIS Institute <sup>7</sup>KE:SAI <sup>\*</sup>Equal contribution

merve.kocabas@student.uni-tuebingen.de a.geiger@uni-tuebingen.de {bburns46,gauenk,sfk}@gatech.edu

## ABSTRACT

Diffusion models have become strong priors for solving inverse problems. However, nonlinear inverse problems such as phase retrieval remain challenging when multiple measurement-consistent solutions are distinguishable only by the prior; we call such measurements ambiguous. Existing diffusion-based samplers rely on local approximations during reconstruction. For an ambiguous measurement, these approximations can allow measurement guidance to favor a measurementconsistent mode that is close to a prior-supported mode but is itself not supported by the prior. Evaluation often masks this type of failure by reporting the best-of-k runs, where the best is selected by comparison to the ground truth. We introduce Prior-Consistent DAPS (PC-DAPS), a variant of Decoupled Annealing Posterior Sampling (DAPS) designed to address this failure case by maintaining multiple candidate solutions within a single sampling run. As annealing noise decreases, PC-DAPS prunes and replaces weaker candidates according to their prior consistency, without any ground-truth access. PC-DAPS reduces the incidence of priorinconsistent samples and improves sample quality on nonlinear inverse problems while preserving sample quality and diversity on linear inverse problems.<sup>†</sup>

## 1 INTRODUCTION

![](images/636dd51593e680f09ae7244cd39e4e8299ba14c328b40c97b0cc24182dafdf68.jpg)  
Figure 1: Measurement consistency is not sufficient for reconstruction. A single Fourier magnitude measurement is shared by rotations, translations, and color-channel inversions of an image. Existing solvers may return these prior-inconsistent modes, while PC-DAPS selects reconstructions supported by the learned image prior. Each sample was selected randomly, either from four independent runs for other methods or from the four particles generated in a single PC-DAPS run.

Inverse problems arise across science and engineering whenever an unknown signal must be recovered from indirect measurements, including image restoration (Bertero et al., 2021), medical imaging (Song et al., 2022), and astronomical imaging (Event Horizon Telescope Collaboration, 2019). Given an observed measurement y generated through a known but potentially noisy measurement process, the goal of the inverse problem is to recover the original signal x. For underdetermined problems, in which many different signals x may give rise to the same measurement $y ,$ Bayesian imaging defines the posterior distribution as $p ( x \mid y ) \propto p ( y \mid x ) p ( x )$ , where the likelihood $p ( y \mid x )$ captures measurement consistency of a candidate image x and the prior $p ( x )$ captures which candidate reconstructions are plausible natural signals. The goal of posterior sampling is thus to generate candidate reconstructions that are simultaneously consistent with the measurement and the prior.

In Fourier phase retrieval, for example, multiple images can produce identical Fourier magnitude measurements (Fienup, 1982). These measurement-equivalent solutions therefore have the same likelihood, but can have very different posterior probabilities depending on their consistency with the image prior. In particular, we observe that existing phase-retrieval solvers applied to face images sometimes return a rotated or color-inverted face that is consistent with the measurement but implausible under a natural-image prior concentrated on upright faces. The challenge is therefore not simply to find measurement-consistent reconstructions, but to sample the measurement-consistent solutions supported by the prior.

Diffusion models provide strong image priors for posterior sampling (Ho et al., 2020; Song et al., 2021). At each noise level, a diffusion model implicitly defines a denoiser that maps a noisy sample x to an estimate of the corresponding clean image; this diffusion denoiser can be used to incorporate the prior $p ( x )$ during posterior sampling. Diffusion-based inverse solvers apply this prior information along a sampling trajectory, i.e., the sequence of intermediate states that evolves from a noisy sample toward a reconstruction. Existing methods primarily either tilt the unconditional reverse diffusion process using an approximate likelihood gradient (Zhang et al., 2025; Chung et al., 2023; Song et al., 2023; Boys et al., 2024) or alternate prior and measurement updates across noise levels (Wu et al., 2024; Xu & Chi, 2024; Chen et al., 2026; Ahmed & Makin, 2026). In these methods, the prior information available at each noise level is represented through a local denoising approximation centered on a single clean-image estimate, rather than explicitly maintaining multiple competing prior modes at intermediate timesteps.

When the measurement is ambiguous, as in the case of phase retrieval and other underdetermined inverse problems, following a sampling trajectory can lead to premature mode selection. At high noise levels, multiple prior modes may remain plausible, but the trajectory can only move towards one of them. Once a trajectory selects a particular prior mode, subsequent refinement primarily improves measurement consistency in the local neighborhood of that mode rather than considering more distant alternatives. This becomes problematic when (i) the selected prior mode is not consistent with the measurement, and (ii) its nearest measurement-consistent mode is not supported by the prior. Trajectory-based samplers can prematurely commit to this pair of modes, ultimately resulting in hallucinatory sampled reconstructions that are not supported by the true posterior or even the true prior. Best-of-k evaluation partially masks this issue by generating multiple independent reconstructions and selecting the one closest to the ground truth (Chung et al., 2023; Zhang et al., 2025). However, this selection uses information unavailable at test time and does not ensure that the produced samples are prior-consistent. Particle-based approaches maintain multiple candidates at intermediate timesteps (Cardoso et al., 2024; Dou & Song, 2024; Wu et al., 2023), but have primarily focused on linear inverse problems and particle diversity rather than prior-consistent sampling.

To address this issue, we introduce Prior-Consistent DAPS (PC-DAPS), which moves sample selection inside the sampler and removes the need for ground-truth-based best-of-k evaluation. PC-DAPS maintains a bag of K particles annealed under the same measurement, and models the denoiser posterior $p ( x _ { 0 } \mid x _ { t } )$ with a K-component Gaussian mixture rather than a unimodal estimate. Early in sampling, multiple candidates are kept while the noise level is high and the prior is still weakly informative. As the noise decreases during annealing, particles with low prior consistency are pruned and replaced by surviving candidates that are more prior-consistent. PC-DAPS thus delays mode selection until the prior becomes more informative, performing within a single sampling run (and without ground-truth access) what best-of-k does only after k independent sampling runs.

We evaluate PC-DAPS on FFHQ (Karras et al., 2019) and ImageNet (Deng et al., 2009) across three nonlinear and seven linear tasks, including the eight tasks reported in DAPS (Zhang et al., 2025) as well as high-pass filtering (Thaker et al., 2025) and 16× super-resolution. We report PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018) under both single-sample and best-of-k evaluation. We additionally compare the rate of prior-inconsistent hallucinations against independent DAPS runs. PC-DAPS reduces the incidence of prior-inconsistent samples and improves performance on ambiguous nonlinear problems while maintaining image quality and diversity on linear inverse problems. Our contributions are as follows:

• We propose PC-DAPS, which maintains a bag of particles, models the denoiser posterior with a Gaussian mixture, and performs prior-based pruning without access to ground truth.

• We evaluate PC-DAPS across seven linear and three nonlinear inverse problems on FFHQ and ImageNet, showing fewer prior-inconsistent reconstructions and improved nonlinear reconstruc tion quality while preserving performance on linear tasks.

## 2 RELATED WORK

Likelihood-guided diffusion sampling. Given a diffusion prior that has learned the unconditional score $\nabla _ { x _ { t } } \log { p ( x _ { t } ) }$ , we can decompose the conditional score (after observing measurement y) as $\nabla _ { x _ { t } }$ log $p ( x _ { t } \ \mid \ j ) = \nabla _ { x _ { t } } \log p ( x _ { t } ) \bar { \ } + \nabla _ { x _ { t } } \log p ( y \mid \ x _ { t } )$ . The likelihood term $\nabla _ { x _ { t } } \log p ( y \mid x _ { t } )$ is intractable, so a family of methods approximates it to steer the reverse diffusion process. In particular, DPS (Chung et al., 2023) evaluates the likelihood at the one-step posterior mean $p ( \boldsymbol { \bar { y } } \mid \boldsymbol { x } _ { t } ) \approx p ( \boldsymbol { y } \mid \mathbb { E } [ \boldsymbol { x } _ { 0 } \mid \mathbf { \bar { \boldsymbol { x } } } _ { t } ] )$ and backpropagates to compute the score. For linear measurement operators, ΠGDM (Song et al., 2023) and TMPD (Boys et al., 2024) instead propagate Gaussian approximations of $p ( x _ { 0 } \mid x _ { t } )$ , while DDRM (Kawar et al., 2022) utilizes a diagonalization approach. MCG (Chung et al., 2022) adds a manifold-correction term to keep the guided iterate close to the data manifold. Jin et al. (2026) target hallucination inside this family of approaches by stabilizing the prior update, but remain committed to a single trajectory.

Split and annealed posterior sampling. In contrast, variable-splitting methods alternate between prior-only and measurement-only steps. Among methods using a diffusion model for the prior step, PnP-DM (Wu et al., 2024) uses a split Gibbs sampler for the likelihood, while DPnP (Xu & Chi, 2024) uses Metropolis-Hastings and DiffPIR (Zhu et al., 2023) utilizes a proximal step. Finally, DAPS (Zhang et al., 2025) utilizes a repeated denoising-renoising scheme (Section 3.2). RePS (Ahmed & Makin, 2026) adds restarts to that schedule, and DAPS++ (Chen et al., 2026) reports that the DAPS reconstruction is largely driven by measurement consistency. We therefore design PC-DAPS as a modification of DAPS that retains its variable-splitting structure but introduces priorbased selection to discriminate between measurement-consistent reconstructions.

Particle-based posterior sampling. Another group of approaches maintains many particles throughout sampling, and uses sequential Monte Carlo to propagate, reweight, and resample particles to progressively improve measurement consistency (Cardoso et al., 2024; Janati et al., 2025). FPS-SMC (Dou & Song, 2024) casts the reverse sampling process as a state-space model and applies a particle filter. TDS (Wu et al., 2023) uses twisting functions to obtain asymptotic exactness for general measurement models. While particle-based methods are typically concerned with the fidelity of the ensemble in the limit of many particles, we are concerned instead with the probability that a single returned reconstruction is prior-inconsistent, and use particles for the purpose of exploring the posterior landscape.

Diagnosing hallucination and prior misfit. Hallucination has been studied as a property of reconstruction algorithms in inverse problems since before diffusion priors were adopted (Bhadra et al., 2021), and refers to any reconstruction that is inconsistent with either the measurement or the prior. Recent work has sought to quantify hallucination specifically for diffusion-based solvers (Jin et al., 2026). At the level of the model rather than the individual sample, DiME (Wang & Bouman, 2026) estimates the evidence of a diffusion prior by integrating over the time-marginals of a posterior sampler, and can be run alongside DAPS to detect prior misfit. Closest to our framing, Burns & Fridovich-Keil (2026) adopt a finite-sample perspective on reverse diffusion dynamics and show that widely used guidance approximations misestimate posterior spread at intermediate noise levels. This misestimation produces both prior modes absent from the true posterior and likelihood (measurement-consistent) modes unsupported by the prior, which is the specific failure that our PC-DAPS method aims to address. Benchmarks whose posteriors are analytically tractable provide a complementary check on the uncertainty a solver reports (Crafts & Villa, 2025; Zach et al., 2026; Zheng et al., 2025). All of these existing hallucination diagnostics are applied after sampling; PC-DAPS utilizes prior support during sampling to decide which reconstructions survive, thereby reducing the incidence of prior-inconsistent hallucinations.

## 3 PRELIMINARIES

## 3.1 BAYESIAN INVERSE PROBLEMS

We consider Bayesian inverse problems of the form:

$$
y = \mathcal { A } ( x ) + n , \qquad n \sim \mathcal { N } ( 0 , \beta _ { y } ^ { 2 } I ) ,\tag{1}
$$

where $x \in \mathbb { R } ^ { d }$ is the unknown signal of interest, $\mathcal { A } : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ is a known measurement operator that is deterministic but may be linear or nonlinear, $n \in \mathbb { R } ^ { m }$ is unknown additive Gaussian noise, and $y \in \mathbb { R } ^ { m }$ is the observed measurement. Given a prior distribution $p ( x )$ , Bayes’ rule gives the posterior distribution over likely signals x consistent with a provided measurement y:

$$
p ( x \mid y ) \propto p ( y \mid x ) p ( x ) , \qquad p ( y \mid x ) = \mathcal { N } \big ( y ; \mathcal { A } ( x ) , \beta _ { y } ^ { 2 } I \big ) .\tag{2}
$$

We are particularly interested in forward models where the set of measurement-consistent signals

$$
\mathcal { M } ( y ) = \left\{ x \in \mathbb { R } ^ { d } : \| A ( x ) - y \| \leq \delta \right\}\tag{3}
$$

contains multiple candidate reconstructions which are well-separated in signal space. In this context, two distinct signals in $\mathcal M ( y )$ can be assigned equal likelihoods $p ( y \mid x )$ but have drastically different posterior probabilities $p ( x \mid y )$ , as one image might be unrealistic $( \mathrm { i . e . }$ , significantly lower prior probability $p ( x ) )$ . A common failure case of existing reconstruction methods is to sample both signals with roughly equal frequency, thereby misrepresenting their posterior probabilities as relatively equal.

For example, in Fourier phase retrieval, two images which are identical up to rotation, translation, and/or color channel sign appear visually quite distinct but produce identical Fourier magnitude measurements (Fienup, 1982). On structured face images, a solver can return a 180<sup>◦</sup>-rotated or color-inverted (e.g., blue) face that fits the observed Fourier measurement despite being inconsistent with a prior distribution supported only on the set of upright faces. We refer to such reconstructions as prior-inconsistent. Prior-inconsistent reconstructions commonly appear across classical phaseretrieval methods (Fienup, 1982), deep-learning approaches (Metzler et al., 2018; Is¸il & Oktem, 2025), and diffusion-based inverse-problem solvers (Xu & Chi, 2024; Zhang et al., 2025). Our goal is to reduce the frequency of prior-inconsistent solutions while retaining quality and diversity of measurement- and prior-consistent samples.

## 3.2 DECOUPLED ANNEALING POSTERIOR SAMPLING (DAPS)

To develop a sampler which produces fewer prior-inconsistent samples, we build on existing approaches by (i) identifying the root causes that produce the inconsistent samples and (ii) making adjustments to reduce the incidence of these samples. Specifically, we build on DAPS (Zhang et al., 2025), a diffusion-based variable-splitting approach which repeatedly (i) denoises to a clean image via the reverse diffusion process, (ii) improves the clean image’s likelihood consistency via MCMC (e.g., Langevin dynamics (Roberts & Tweedie, 1996)), and (iii) renoises the MCMC output via the forward diffusion process. By adding less noise back than was originally present, DAPS aims to iteratively denoise towards a measurement-consistent clean sample.

Mathematically, DAPS utilizes the fact that given an intermediate image $x _ { t } \sim p ( x _ { t } \mid y )$ , drawing:

$$
x _ { 0 } \sim p ( x _ { 0 } \mid x _ { t } , y ) , \qquad x _ { t - 1 } \sim { \mathcal { N } } { \bigl ( } x _ { 0 } , \sigma _ { t - 1 } ^ { 2 } I { \bigr ) } ,\tag{4}
$$

produces $x _ { t - 1 } \sim p ( x _ { t - 1 } \mid y )$ . To sample from the guided denoising distribution $p ( x _ { 0 } \mid x _ { t } , y )$ observe that y and $x _ { t }$ are independent conditioned on $x _ { 0 } ;$ thus, by Bayes’ rule we can factorize:

$$
p ( x _ { 0 } \mid x _ { t } , y ) \propto p ( y \mid x _ { 0 } ) p ( x _ { 0 } \mid x _ { t } ) .\tag{5}
$$

While $p ( y \mid x _ { 0 } )$ is induced by Equation (1), the denoising distribution $p ( x _ { 0 } \mid x _ { t } )$ is unknown and intractable to estimate via sampling. DAPS utilizes a Gaussian approximation for this distribution:

$$
p ( \boldsymbol { x } _ { 0 } \mid \boldsymbol { x } _ { t } ) \approx \mathcal { N } \big ( \boldsymbol { x } _ { 0 } ; \hat { \boldsymbol { x } } _ { 0 } ( \boldsymbol { x } _ { t } ) , r _ { t } ^ { 2 } I \big ) ,\tag{6}
$$

![](images/796ecf0884f57005891b6bad7dc0654513f9f338b1f020a68e1acd2e4bccbfed.jpg)  
majority posterior mode x minority posterior mode x DAPS-hallucinated mode $x _ { 3 }$  
Figure 2: Premature mode selection. We consider a nonlinear inverse problem with quadratic A and a two-atom prior; details are provided in Appendix A.2. We select y such that one atom is likelihood-consistent while the other is likelihood-inconsistent but neighboring a likelihoodconsistent prior-inconsistent mode. The exact posterior, DAPS, and PC-DAPS are shown as distributions across noise levels $\sigma .$ . The exact posterior assigns positive weight to both atoms, with majority weight on the likelihood-consistent atom. However, DAPS assigns majority mass to the prior-inconsistent likelihood mode, as highlighted in the last panel, which is not present in the true posterior. PC-DAPS’s selection algorithm prunes this mode, sampling only from the posteriorconsistent atom. While neither DAPS nor PC-DAPS samples from the likelihood-inconsistent atom with adequate probability, PC-DAPS eliminates the hallucinated prior-inconsistent mode produced by DAPS.

where $\hat { x } _ { 0 } ( x _ { t } )$ is an estimate of $\mathbb { E } [ x _ { 0 } \mid x _ { t } ]$ and $r _ { t }$ is set heuristically. Observe that the (Stein) score w.r.t. $x _ { 0 }$ of the approximated right-hand side of Equation (5) is available in closed form, allowing for sampling via the unadjusted Langevin algorithm with step size $\eta _ { t }$ from initialization $\hat { x } _ { 0 } ( x _ { t } ) \colon$

$$
\hat { x } _ { 0 } ^ { ( j + 1 ) } \gets \hat { x } _ { 0 } ^ { ( j ) } + \eta _ { t } \big ( \nabla _ { \hat { x } _ { 0 } } \log p ( \hat { x } _ { 0 } ^ { ( j ) } \mid x _ { t } ) + \nabla _ { \hat { x } _ { 0 } } \log p ( y \mid \hat { x } _ { 0 } ^ { ( j ) } ) \big ) + \sqrt { 2 \eta _ { t } } \epsilon ^ { ( j ) } , \quad \epsilon ^ { ( j ) } \sim \mathcal { N } ( 0 , I ) .\tag{7}
$$

Once the Langevin algorithm terminates (after a user-prescribed number of iterations $N )$ , the result is renoised using the EDM parameterization (Karras et al., 2022) of the forward diffusion process $x _ { t - 1 } = \hat { x } _ { 0 } ^ { ( N ) } + \sigma _ { t - 1 } \epsilon$ with $\epsilon \sim \mathcal { N } ( 0 , I )$ . Renoising in DAPS is introduced to counteract the locality of Langevin refinement: because consecutive Langevin updates are local, the samples may fail to explore alternative prior modes distant from $\hat { x } _ { 0 } ( x _ { t } )$ The renoising-denoising update intends to allow particles to make larger moves per iteration and to investigate alternative prior modes.

However, the DAPS update rule still produces the prior-inconsistent samples described in Section 3.1, as shown in Figure 1. Observe that the DAPS procedure for approximately sampling from $p ( x _ { 0 } \mid x _ { t } , y )$ in effect produces a prior-consistent sample near $\mathbb { E } [ x _ { 0 } \mid x _ { t } ]$ and samples from likelihood-consistent regions of its neighborhood. In contexts where $x _ { t }$ selects a prior-consistent sample that is likelihood-inconsistent, yet there exists a nearby region of signal space consistent with $y ,$ Equation (7) may sample from that region even if it is prior-inconsistent, as the Gaussian approximation to $p ( x _ { 0 } \mid x _ { t } )$ merely discourages the samples from deviating far in signal space from $\hat { x } _ { 0 } ( x _ { t } )$ . We observe that $x _ { 0 }$ renoises to an $x _ { t }$ that subsequently denoises to the same region with high probability, causing the particle to become trapped in a prior-inconsistent but measurementconsistent mode. We refer to this phenomenon as premature mode selection because the sample x<sub>t</sub> prematurely commits to only sampling $x _ { 0 }$ from a particular prior mode, without being able to adequately decommit even if that prior mode is not present in the posterior.

Figure 2 illustrates this phenomenon. Here, the prior places mass on two candidate solutions, $x _ { 1 }$ and $x _ { 2 } ,$ , while the measurement $y$ is explained by $x _ { 1 }$ and a third point $x _ { 3 }$ unsupported by the prior. In analogy to Fourier phase retrieval, one can think of $x _ { 1 }$ as the desired reconstruction, e.g., an upright, non-blue face that is both consistent with the measurement and supported by the image prior, $x _ { 2 }$ as a realistic face inconsistent with the particular measurement, and $x _ { 3 }$ as a structurally implausible transformation of the desired image $x _ { 1 }$ , such as a rotated or blue face. The exact posterior should favor the $x _ { 1 }$ mode while assigning zero mass to the prior-inconsistent $x _ { 3 }$ . However, DAPS samples from $x _ { 3 } \ ' _ { \mathbf { S } }$ neighborhood by (i) sampling $x _ { 2 }$ from the prior and (ii) sampling $x _ { 3 }$ from $x _ { 2 } \mathrm { ^ { * } s }$ local likelihood. Once a particle enters this locally attractive likelihood-consistent region, subsequent renoising and Langevin updates become trapped, resulting in premature mode selection.

This toy example motivates the design of PC-DAPS, which maintains multiple candidate measurement-consistent modes to identify and correct trapped prior-inconsistent samples. As shown in Figure 2, PC-DAPS places its mass only on the prior-supported and measurement-consistent mode at $x _ { 1 }$ and avoids sampling from the prior-inconsistent, measurement-consistent region near $x _ { 3 }$ . We describe how PC-DAPS identifies and removes such prior-inconsistent particles in the next section.

## 4 METHOD

To address the presence of prior-inconsistent samples, a first approach might be to produce multiple samples via DAPS and discard any which are prior-inconsistent. There are two main shortcomings of this approach: (i) the effective sample size of this sampling method is suboptimal, as substantial computational resources are used to produce samples that are then thrown out, and (ii) it is unclear how to select which samples to discard without having access to the ground truth signal or posterior.

To address these issues, we introduce PC-DAPS, a procedure for correcting prematurely committed particles during the sampling process without requiring access to the ground truth image or to any pre-trained prior-inconsistency classifier. PC-DAPS maintains a bag of K particles, each of which is denoised using the same annealing schedule and pretrained denoiser, but uses different initializations and random noise in its Langevin and renoising updates. If no additional modifications are made to DAPS, the K particles will denoise independently, and can be partitioned into (nonempty) sets $\mathrm { o f } \ ^ { \mathrm { 6 4 } }$ prematurely committed particles and “good” particles that are posterior-consistent. The core idea of PC-DAPS is to improve DAPS’s existing renoising strategy by identifying prematurely committed particles during sampling and moving them to more promising modes discovered by other particles in the bag. Additionally, PC-DAPS utilizes the particles to refine additional design decisions made by DAPS, such as by replacing the unimodal Gaussian approximation to the denoising distribution with an adaptive Gaussian mixture model (GMM) approximation. A toy example illustrating the benefit of this Gaussian mixture approximation is shown in Appendix A.3.

What remains is to describe how PC-DAPS (i) identifies when particles commit prematurely, (ii) identifies which particles are problematic, (iii) corrects the identified particles, and (iv) constructs the multimodal denoising distribution approximation from the bag. PC-DAPS makes two core changes to DAPS, utilizing K particles annealed in parallel: a pruning technique to correct particles trapped in prior-inconsistent likelihood modes in tasks with likelihood ambiguity (Section 4.1), and a multimodal denoising distribution approximation estimated from the diversity of the particles (Section 4.2). Additional implementation details and ablations are provided in Appendix A.4, such as omitting the final Langevin refinement and adaptive ODE stepping from high to low noise to improve reconstruction quality while reducing computational cost (Appendix A.5). The full algorithm (Algorithm 1) can be found in Appendix A.1.

## 4.1 PRUNING SCHEME

PC-DAPS begins by initializing a bag of K particles $\boldsymbol { B } = \{ x _ { t _ { \mathrm { m a x } } } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ from the initial distribution of the reverse diffusion process, and denoises the particles as prescribed by DAPS. On tasks where $\mathcal M ( y )$ (Equation (3)) contains multiple clearly separated candidates (e.g., for a nonlinear ${ \mathcal { A } } ) ,$ the particles will begin to denoise to genuinely different candidate reconstructions. At the annealing time of the DAPS denoising process where the prior first becomes discriminative, PC-DAPS invokes a one-shot consensus check to identify prior-inconsistent particles.

To determine when the check should occur, we test the diversity of the bag at each iteration. Define:

$$
\bar { r } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left\| \mathcal { A } ( { x } ^ { ( k ) } ) - { y } \right\| ^ { 2 } , \qquad r ^ { \star } = \left\| \mathcal { A } ( \bar { x } ) - { y } \right\| ^ { 2 } , \qquad \bar { x } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { x } ^ { ( k ) } .\tag{8}
$$

Observe that, for linear ${ \mathcal { A } } ,$ Jensen’s inequality forces $\bar { r } \geq r ^ { \star }$ by convexity of the squared norm. Using this convexity intuition, for nonlinear A the condition that $\bar { r } ~ < ~ r ^ { \star }$ indicates that the bag occupies multiple wells of the likelihood potential, indicating that mode-level pruning is meaningful.

To identify which particles require correcting, we partition the bag into clusters ${ \mathcal { C } } _ { m }$ and evaluate each by its average prior consistency (obtained via Tweedie’s formula) over several noise scales S:

$$
s ( \mathcal { C } _ { m } ) = - \frac { 1 } { | \mathcal { C } _ { m } | | S | } \sum _ { x \in \mathcal { C } _ { m } } \sum _ { \sigma \in \mathcal { S } } \bigl \| x - D _ { \theta } ( x + \sigma \epsilon ; \sigma ) \bigr \| ^ { 2 } , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) .\tag{9}
$$

The premise of this criterion is that a sample lying near the (prior) data manifold denoises with small (Stein) score at every noise scale. If a cluster’s prior-consistency score $s ( \mathcal { C } _ { m } )$ exceeds a threshold, its particles are unaltered. Otherwise, its particles are replaced randomly by copies of particles from clusters scoring above the threshold. Implementation choices including the clustering method, choice of noise scales $s ,$ , and determination of threshold are discussed in Appendix A.4.

## 4.2 GAUSSIAN MIXTURE APPROXIMATION OF THE DENOISER POSTERIOR

An additional benefit of maintaining a bag of particles is that we can improve upon DAPS’s unimodal approximation in Equation (6) by selecting a multimodal distribution informed by the bag’s diversity. We propose replacing the DAPS Gaussian approximation with a Gaussian mixture. Denote the clean-signal estimate of the k-th particle by $\bar { \boldsymbol { x } } _ { 0 } ^ { ( k ) } = D _ { \theta } \big ( \boldsymbol { x } _ { t } ^ { ( k ) } ; \boldsymbol { \sigma } _ { t } \big )$ . Before the pruning described in Section 4.1 activates, we use a uniformly-weighted mixture of Gaussians centered at the clean estimates:

$$
q ( x _ { 0 } \mid x _ { t } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { N } \Big ( x _ { 0 } ; \tilde { x } _ { 0 } ^ { ( k ) } , \mathrm { d i a g } \left( \mathrm { V a r } _ { \mathcal { B } } \big ( \tilde { x } _ { 0 } ^ { ( k ) } \big ) \right) \Big ) .\tag{10}
$$

After pruning, we maintain a mode for each unpruned cluster, where the corresponding component parameters are given by the cluster’s empirical weight, mean, and diagonal covariance:

$$
\boldsymbol { q } ( x _ { 0 } \mid x _ { t } ) = \sum _ { m = 1 } ^ { M } \pi _ { m } \boldsymbol { \mathcal { N } } \big ( x _ { 0 } ; \mu _ { m } , \Sigma _ { m } \big ) ,\tag{11}
$$

$$
\pi _ { m } = \frac { | \mathcal { C } _ { m } | } { K } , \qquad \mu _ { m } = \frac { 1 } { | \mathcal { C } _ { m } | } \sum _ { x ^ { ( k ) } \in \mathcal { C } _ { m } } \tilde { x } _ { 0 } ^ { ( k ) } , \qquad \Sigma _ { m } = \mathrm { d i a g } \Big ( \mathrm { V a r } _ { \mathcal { C } _ { m } } \big ( \tilde { x } _ { 0 } ^ { ( k ) } \big ) \Big ) .\tag{12}
$$

When a cluster contains only a single particle, its component uses the full bag’s diagonal covariance.

Unlike DAPS, which centers the denoising approximation at the point estimate $\hat { x } _ { 0 } ( x _ { t } )$ , PC-DAPS shares the bag-mean center across all particles in a mode, giving a variance-reduced prior estimate that we find improves reconstruction quality on linear reconstruction tasks. Furthermore, when the full bag lies in a single cluster, the mixture reduces to a single Gaussian centered on the bag’s average Tweedie estimate with data-dependent covariance. This contrasts with the heuristic, geometryindependent covariance utilized by DAPS.

Observe that the score of the GMM denoising approximation $q ( x _ { 0 } \mid x _ { t } )$ is accessible in closed form:

$$
\nabla _ { x _ { 0 } } \log { q ( x _ { 0 } \mid x _ { t } ) } = \sum _ { m = 1 } ^ { M } q _ { m } ( x _ { 0 } ) \Sigma _ { m } ^ { - 1 } \bigl ( \mu _ { m } - x _ { 0 } \bigr ) ,\tag{13}
$$

with weights $\begin{array} { r } { q _ { m } ( x _ { 0 } ) = \pi _ { m } \mathcal { N } ( x _ { 0 } ; \mu _ { m } , \Sigma _ { m } ) / \sum _ { m ^ { \prime } } \pi _ { m ^ { \prime } } \mathcal { N } ( x _ { 0 } ; \mu _ { m ^ { \prime } } , \Sigma _ { m ^ { \prime } } ) } \end{array}$ , and thus can be utilized in the DAPS Langevin update introduced in Equation (7). Observe that when the weights $q _ { m }$ are approximately one-hot, this update approximately reduces to a data-dependent Gaussian. Recalling that the DAPS Gaussian approximation disincentivizes the Langevin dynamics from deviating far from the point estimate $\hat { x } _ { 0 } ( x _ { t } )$ , the PC-DAPS Gaussian approximation discourages the Langevin dynamics from deviating far from the prior mode identified by a collective cluster of the bag, as demonstrated in Figure 4 in Appendix A.3.

## 5 RESULTS

Main results. Table 1 shows that while PC-DAPS is not uniformly superior to all baselines, it performs competitively across all experiments and often achieves the highest sample quality, with the most substantial improvements in phase retrieval. In particular, the phase-retrieval results for

Table 1: Quantitative evaluation on FFHQ and ImageNet. Comparison of methods on FFHQ (left) and ImageNet (right) across seven linear and three nonlinear inverse problems (defined in Appendix A.6) using PSNR, SSIM, and LPIPS, evaluated over 100 test images. For the nonlinear problems, mean ± standard deviation over 100 images is reported to highlight variation across samples. PC-DAPS produces 4 samples for each input, so the metrics reported for PC-DAPS are averaged over these 4 samples. For phase retrieval, for all methods other than PC-DAPS, we report oracle results from best-of-4 runs. We indicate the best and second-best results for each experiment with bold and underline annotations. We use a noisy measurement with noise level $\beta _ { y } = 0 . 0 5$ in all tasks. Baseline implementation details are provided in Appendix A.6.1.
<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=5>FFHQ                             ImageNetPSNR (↑)   SSIM (↑)   LPIPS (↓)  PSNR (↑)   SSIM (↑)   LPIPS (↓)</td></tr><tr><td rowspan=3 colspan=1>Super-resolution 4×</td><td rowspan=3 colspan=1>PC-DAPSDAPSDDNMDPSFPS-SMC</td><td rowspan=2 colspan=4>30.25      0.841       0.16629.36      0.782       0.173</td><td rowspan=1 colspan=1>26.44      0.714       0.245</td></tr><tr><td rowspan=1 colspan=1>25.51      0.639       0.281</td></tr><tr><td rowspan=1 colspan=4>28.01      0.797      0.19625.86      0.753      0.26928.45      0.812      0.206</td><td rowspan=1 colspan=1>23.92      0.606       0.47221.13      0.489       0.36124.79      0.705       0.311</td></tr><tr><td rowspan=3 colspan=1>Super-resolution 16×</td><td rowspan=3 colspan=1>PC-DAPSDAPSDPSFPS-SMC</td><td rowspan=3 colspan=4>22.21      0.623      0.36822.14      0.597       0.37321.09      0.586      0.57022.09      0.604       0.496</td><td rowspan=1 colspan=1>20.90      0.454       0.53520.35      0.450       0.545</td></tr><tr><td rowspan=1 colspan=1>18.41      0.384       0.592</td></tr><tr><td rowspan=1 colspan=1>18.88      0.423       0.588</td></tr><tr><td rowspan=3 colspan=1>High-pass filter</td><td rowspan=3 colspan=1>PC-DAPSDAPSDPSFGPS</td><td rowspan=1 colspan=4>18.84      0.815       0.155</td><td rowspan=1 colspan=1>17.33      0.778       0.204</td></tr><tr><td rowspan=1 colspan=4>16.85      0.702      0.204</td><td rowspan=1 colspan=1>15.51      0.732       0.217</td></tr><tr><td rowspan=1 colspan=4>4.92      0.437       0.68718.04      0.761       0.184</td><td rowspan=1 colspan=1>12.84      0.606       0.47314.77      0.708       0.229</td></tr><tr><td rowspan=5 colspan=1>Inpaint (box)</td><td rowspan=3 colspan=1>PC-DAPSDAPSDDNM</td><td rowspan=2 colspan=4>24.73      0.825       0.15224.69      0.746      0.166</td><td rowspan=1 colspan=1>21.56      0.738       0.221</td></tr><tr><td rowspan=1 colspan=1>21.12      0.714      0.229</td></tr><tr><td rowspan=1 colspan=4>24.44      0.839      0.233</td><td rowspan=1 colspan=1>21.67      0.746       0.321</td></tr><tr><td rowspan=2 colspan=1>DPSFPS-SMC</td><td rowspan=1 colspan=4>22.51      0.792      0.209</td><td rowspan=1 colspan=1>18.94      0.722       0.257</td></tr><tr><td rowspan=1 colspan=4>24.86      0.823      0.146</td><td rowspan=1 colspan=1>22.16      0.726       0.208</td></tr><tr><td rowspan=5 colspan=1>Inpaint (random)</td><td rowspan=3 colspan=1>PC-DAPSDAPSDDNM</td><td rowspan=1 colspan=4>31.83      0.875       0.121</td><td rowspan=1 colspan=1>28.18      0.781       0.159</td></tr><tr><td rowspan=1 colspan=4>30.73      0.801       0.129</td><td rowspan=1 colspan=1>27.32      0.725       0.189</td></tr><tr><td rowspan=1 colspan=4>29.87      0.816       0.123</td><td rowspan=1 colspan=1>31.14      0.839       0.193</td></tr><tr><td rowspan=1 colspan=1>DPS</td><td rowspan=1 colspan=4>25.46      0.823      0.203</td><td rowspan=1 colspan=1>23.52      0.745       0.297</td></tr><tr><td rowspan=1 colspan=1>FPS-SMC</td><td rowspan=1 colspan=4>28.17      0.824      0.258</td><td rowspan=1 colspan=1>24.55      0.699       0.318</td></tr><tr><td rowspan=5 colspan=1>Gaussian deblurring</td><td rowspan=5 colspan=1>PC-DAPSDAPSDDNMDPSFPS-SMC</td><td rowspan=2 colspan=4>30.35      0.839       0.16429.60      0.787      0.179</td><td rowspan=2 colspan=1>26.37      0.703       0.2500.659       0.267</td></tr><tr><td rowspan=1 colspan=1>25.940.65</td></tr><tr><td rowspan=3 colspan=4>28.18      0.805      0.21425.87      0.764       0.21926.47      0.771       0.251</td><td rowspan=1 colspan=1>0.701       0.281</td></tr><tr><td rowspan=1 colspan=1>20.31      0.598       0.397</td></tr><tr><td rowspan=1 colspan=1>23.86      0.603       0.390</td></tr><tr><td rowspan=3 colspan=1>Motion deblurring</td><td rowspan=3 colspan=1>PC-DAPSDAPSDPSFPS-SMC</td><td rowspan=2 colspan=4>32.13      0.872       0.11931.65      0.837      0.13724.52      0.801       0.246</td><td rowspan=1 colspan=1>29.06      0.800       0.16228.66      0.770       0.175</td></tr><tr><td rowspan=1 colspan=1>18.96      0.629       0.423</td></tr><tr><td rowspan=1 colspan=4>27.31      0.825      0.229</td><td rowspan=1 colspan=1>24.60      0.645       0.329</td></tr><tr><td rowspan=5 colspan=1>Phase retrieval</td><td rowspan=3 colspan=1>PC-DAPSDAPSDPSPnP-DM</td><td rowspan=3 colspan=4> ${ \bf 3 0 . 0 9 } _ { \pm 1 . 7 7 }$    $\mathbf { 0 . 8 3 4 _ { \pm 0 . 0 8 9 } }$    $\mathbf { 0 . 1 5 2 _ { \pm 0 . 0 8 8 } }$  $\underline { { 2 9 . 6 6 } } _ { \pm 4 . 1 7 }$    $\underline { { 0 . 7 7 0 } } _ { \pm 0 . 1 0 0 }$    $\underline { { 0 . 1 7 7 } } _ { \pm 0 . 0 9 0 }$  $1 7 . 6 4 _ { \pm 2 . 9 7 } ^ { - }$    $0 . 4 4 1 _ { \pm 0 . 1 2 9 }$    $0 . 4 1 0 _ { \pm 0 . 0 9 0 } ^ { - }$  $2 1 . 7 0 { \scriptstyle \pm 5 . 5 0 }$    $0 . 5 4 2 _ { \pm 0 . 1 2 2 }$    $0 . 3 9 8 { \scriptstyle \pm 0 . 1 0 1 }$ </td><td rowspan=1 colspan=1> $2 1 . 7 5 _ { \pm 6 . 1 0 }$   $\underline { { 0 . 4 6 9 } } _ { \pm 0 . 1 3 2 }$    $\mathbf { 0 . 3 7 3 _ { \pm 0 . 1 2 8 } }$ </td></tr><tr><td rowspan=1 colspan=1>226 0.387+0.163</td></tr><tr><td rowspan=1 colspan=1> $1 7 . 6 8 _ { \pm 6 . 7 0 }$    $0 . 4 5 8 { \scriptstyle \pm 0 . 1 9 1 }$    $0 . 4 9 9 { \scriptstyle \pm 0 . 1 0 1 }$ </td></tr><tr><td rowspan=1 colspan=1>DPnPRED-Diff</td><td rowspan=1 colspan=4> $2 0 . 3 9 { \scriptstyle \pm 6 . 1 5 }$    $0 . 5 9 2 _ { \pm 0 . 1 3 9 }$    $0 . 3 9 2 _ { \pm 0 . 1 7 3 }$  $1 5 . 7 2 _ { \pm 4 . 4 1 }$    $0 . 4 0 1 { \scriptstyle \pm 0 . 1 9 1 }$    $0 . 5 8 8 { \scriptstyle \pm 0 . 0 8 9 }$ </td><td rowspan=1 colspan=1> $1 6 . 7 4 { \scriptstyle \pm 3 . 2 1 }$    $\mathbf { 0 . 4 8 7 _ { \pm 0 . 1 9 6 } }$    $0 . 5 3 5 { \scriptstyle \pm 0 . 1 0 7 }$  $1 5 . 1 1 { \scriptstyle \pm 3 . 6 8 }$    $0 . 3 8 9 { \scriptstyle \pm 0 . 0 5 8 }$    $0 . 5 2 9 { \scriptstyle \pm 0 . 1 2 7 }$ </td></tr><tr><td rowspan=1 colspan=1>HIO</td><td rowspan=1 colspan=4> $1 3 . 5 3 { \scriptstyle \pm 2 . 5 0 }$    $0 . 3 5 9 { \scriptstyle \pm 0 . 0 9 3 }$    $0 . 7 2 6 { \scriptstyle \pm 0 . 0 6 8 }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=1>Nonlinear deblur</td><td rowspan=4 colspan=1>PC-DAPSDAPSDPSRED-Diff</td><td rowspan=3 colspan=4> $2 9 . 0 4 _ { \pm 0 . 8 2 }$    $\mathbf { 0 . 8 0 4 _ { \pm 0 . 0 1 7 } }$    $\underline { { 0 . 1 8 0 } } _ { \pm 0 . 0 2 3 }$  $2 8 . 4 9 { \scriptstyle \pm 1 . 5 2 }$    $0 . 7 6 2 _ { \pm 0 . 0 3 1 }$    $0 . 1 9 1 { \scriptstyle \pm 0 . 0 3 1 }$  $2 3 . 3 9 { \scriptstyle \pm 2 . 0 1 }$                 $0 . 2 7 8 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td rowspan=1 colspan=1> $2 7 . 6 3 { \scriptstyle \pm 2 . 5 8 }$    $\underline { { 0 . 7 3 9 } } _ { \pm 0 . 0 5 3 }$    $\mathbf { 0 . 2 1 1 _ { \pm 0 . 0 3 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>031</td><td rowspan=1 colspan=1> $2 7 . 2 5 { \scriptstyle \pm 3 . 1 8 }$    $0 . 7 1 7 _ { \pm 0 . 0 8 9 }$    $\underline { { 0 . 2 1 3 } } _ { \pm 0 . 0 5 5 }$ </td></tr><tr><td rowspan=1 colspan=2>01 0.623±0.082 0.278±</td><td rowspan=1 colspan=2>8±0.060</td><td rowspan=1 colspan=1> $2 2 . 4 9 { \scriptstyle \pm 3 . 2 0 }$    $0 . 5 9 1 { \scriptstyle \pm 0 . 1 0 1 }$    $0 . 3 0 6 _ { \pm 0 . 0 8 1 }$ </td></tr><tr><td rowspan=1 colspan=4> ${ \bf 3 0 . 7 2 { \scriptstyle \pm 0 . 5 6 } }$    $\underline { { 0 . 7 9 3 } } _ { \pm 0 . 0 2 9 }$    $\mathbf { 0 . 1 6 2 _ { \pm 0 . 0 3 2 } }$ </td><td rowspan=1 colspan=1> ${ \bf 2 9 . 9 1 { \bf _ { \pm 1 . 3 6 } } }$    $\mathbf { 0 . 7 5 1 { \scriptstyle \pm 0 . 0 2 4 } }$    $0 . 2 1 4 { \scriptstyle \pm 0 . 0 8 0 }$ </td></tr><tr><td rowspan=4 colspan=1>High dynamic range</td><td rowspan=4 colspan=1>PC-DAPSDAPSDPSRED-Diff</td><td rowspan=4 colspan=4> $2 8 . 2 5 _ { \pm 1 . 1 7 }$    $\mathbf { 0 . 8 3 9 _ { \pm 0 . 0 2 0 } }$    $\mathbf { 0 . 1 6 3 _ { \pm 0 . 0 3 9 } }$  $\underline { { 2 7 . 6 7 } } \pm 3 . 5 7$    $\mathbf { 0 . 8 3 9 _ { \pm 0 . 0 7 8 } }$    $\underline { { 0 . 1 6 5 } } _ { \pm 0 . 0 6 9 }$  $2 2 . 7 3 _ { \pm 6 . 0 7 }$    $\underline { { 0 . 5 9 1 } } _ { \pm 0 . 1 4 1 }$    $0 . 2 6 4 _ { \pm 0 . 1 5 6 }$  $2 1 . 9 4 _ { \pm 3 . 5 2 }$    $0 . 5 0 8 _ { \pm 0 . 0 8 1 }$    $0 . 2 6 1 _ { \pm 0 . 0 9 1 }$ </td><td rowspan=1 colspan=1> ${ \bf 2 6 . 0 1 _ { \pm 2 . 7 7 } }$    $\mathbf { 0 . 8 2 2 _ { \pm 0 . 0 9 9 } }$    $\mathbf { 0 . 1 7 3 _ { \pm 0 . 0 9 4 } }$ </td></tr><tr><td rowspan=1 colspan=1> $\underline { { 2 5 . 8 0 } } _ { \pm 4 . 3 7 }$    $\underline { { 0 . 8 1 9 } } _ { \pm 0 . 1 1 3 }$    $\underline { { 0 . 1 7 7 } } _ { \pm 0 . 1 0 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $1 9 . 2 3 _ { \pm 2 . 5 2 }$    $0 . 5 8 2 _ { \pm 0 . 0 8 2 }$    $0 . 5 0 3 _ { \pm 0 . 1 0 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $2 1 . 8 8 _ { \pm 5 . 7 7 }$    $0 . 5 9 8 _ { \pm 0 . 0 9 7 }$    $0 . 2 8 1 _ { \pm 0 . 1 9 1 }$ </td></tr></table>

PC-DAPS are competitive across metrics despite its handicap in evaluation protocol; for PC-DAPS we average metrics across the four samples it generates in a single run, whereas all baselines are evaluated on only the best sample (chosen using ground-truth knowledge) generated across four independent runs. Moreover, despite it’s prune-and-replace strategy, we observe that the samples generated within a single PC-DAPS bag typically still contain a similar level of diversity as samples generated by independent DAPS runs, as shown in Figure 3 and Appendix A.7.

Table 2: Phase-retrieval outcome frequencies for DAPS and PC-DAPS on FFHQ and ImageNet. Individual run outcomes classified as Faithful (correctly reconstructed), Upsidedown/color-inverted (converged but with rotation or color ambiguities), or Failed (diverged or noisy output). Unrecoverable denotes instances where all 4 runs or particles within a bag are unfaithful, yielding no reliable sample. Overall, PC-DAPS substantially reduces upside-down/color-inverted samples while increasing faithful reconstructions. The performance gap between datasets highlights the inherent difficulty of phase retrieval on highly diverse, unaligned images (ImageNet) compared to more structured prior domains (FFHQ).  
![](images/64a299292ad472d54057ba224305a17ed31324f46744f050bbdb10948e6992df.jpg)

<table><tr><td>Dataset</td><td>Outcome</td><td></td><td>DAPS (4 runs) PC-DAPS (bag of 4)</td></tr><tr><td rowspan="4">FFHQ</td><td>Faithful ↑</td><td>79.5%</td><td>97.5%</td></tr><tr><td>Upside-down / color-inverted ↓</td><td>14.5%</td><td>1.5%</td></tr><tr><td>Failed ↓</td><td>6.0%</td><td>1.0%</td></tr><tr><td>Unrecoverable runs/bags ↓</td><td>4%</td><td>1%</td></tr><tr><td rowspan="4">ImageNet</td><td>Faithful ↑</td><td>38.5%</td><td>45.0%</td></tr><tr><td>Upside-down / color-inverted ↓</td><td>42.0%</td><td>30.5%</td></tr><tr><td>Failed ↓</td><td>19.5%</td><td>24.5%</td></tr><tr><td>Unrecoverable runs/bags ↓</td><td>25%</td><td>15%</td></tr></table>

Figure 3: PC-DAPS resolves the upside-down orientation inconsistencies seen in standard DAPS best-of-k runs. The diversity rows show that, although PC-DAPS is stable, it preserves subtle variation among the four particles generated within a single run in comparison to the four independent DAPS runs.

Prior-inconsistent hallucinations. Table 1 reports standard metrics to facilitate baseline comparisons. However, the evaluation procedure masks the failure mode PC-DAPS aims to address (measurement-consistent reconstructions that the prior does not support), because the best-of-k evaluation protocol used by baseline methods discards k − 1 of the reconstructed images. On phase retrieval, we instead classify every sample as faithful, upside-down / color-inverted (a measurementconsistent likelihood mode the natural-image prior would down-weight), or failed. Labels are assigned automatically using per-channel PSNR to the ground truth and its rotated version, maximizing over ±1-pixel circular shifts to account for translation ambiguity. A channel is assigned to an orientation only when it is more than 1 dB closer to that orientation; samples with all channels matching the ground truth arefaithful, those matching the rotation or disagreeing across channels are upside-down /color-inverted, and the remainder arefailed. Table 2 reports the resulting frequencies, with examples in Figure 3.

On FFHQ, PC-DAPS lowers the rate of upside-down or color-inverted reconstructions from 14.5% to 1.5% and raises the rate of faithful reconstructions from 79.5% to 97.5%. This addresses the main failure mode PC-DAPS is meant to fix. On ImageNet, the improvements are smaller: the proportion of faithful reconstructions increases from 38.5% to 45.0%, while the proportion of unrecoverable runs/bags (runs/bags where there exists no faithful sample) drops from 25% to 15%, and the rate of upside-down or color-inverted reconstructions decreases from 42.0% to 30.5%. PC-DAPS relies on prior consistency to disambiguate the color and orientation ambiguities in phase-retrieval measurements, but the multi-class ImageNet prior provides weaker disambiguating signals at varying noise scales when compared to the FFHQ prior. When the local geometry of the prior support is more complex, data-manifold-based prior-consistency pruning appears to be less effective.

## 6 DISCUSSION

Motivation, limitations, and future work. Results in the diffusion literature for ambiguous inverse problems are usually reported as best-of-k, with selection using the ground truth. This evaluation approach became common because existing solvers are inconsistent across runs, and a singlesample result would have large variations in quality. We see this instability as a symptom of premature mode selection in the context of measurement ambiguity. We introduce PC-DAPS to utilize the learned prior information to reduce this unwanted variance during sampling without the need for a ground-truth reference. Even in high-dimensional datasets, including FFHQ, our method demonstrates significant improvements due to the local structure of the prior. PC-DAPS obtains a smaller improvement on more structurally diverse datasets, namely ImageNet, because the prior landscape is more complex, exposing a remaining limitation and open problem to be addressed. We encourage future work to further explore our proposed paradigm of leveraging the prior score to assess candidate reconstructions at inference time, and work towards eliminating prior-inconsistent reconstructions even under less-structured priors.

Posterior sampling metrics. Standard metrics like PSNR, SSIM, and LPIPS measure how similar a generated image is to a single reference image. Observe that these evaluation metrics penalize methods that find several plausible solutions that are equally valid under the measurement and prior if the solutions are dissimilar from the fixed reference. Because sample diversity and likelihood consistency are explicit goals of posterior sampling, including in applications such as uncertainty quantification, these sample-quality metrics are not adequate replacements for quality of the sampled posterior. In our work, we report a separate axis of reconstruction quality: prior consistency. We believe this or similar multifaceted metrics should be considered alongside standard fidelity metrics.

## ACKNOWLEDGMENTS

MK is supported by the German Federal Ministry of Education and Research (BMBF) through the DAAD PROMOS program and by the Konrad Zuse School of Excellence in Learning and Intelligent Systems (ELIZA). BAB is supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-2039655. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation. AG is a member of the Machine Learning Cluster of Excellence, EXC 2064/1, project number 390727645.

## REFERENCES

Bilal Ahmed and Joseph G Makin. Solving diffusion inverse problems with restart posterior sampling. In European Conference on Computer Vision (ECCV), pp. 79–96. Springer, 2026.

Mario Bertero, Patrizia Boccacci, and Christine De Mol. Introduction to inverse problems in imaging. CRC press, 2021.

Sayantan Bhadra, Varun A Kelkar, Frank J Brooks, and Mark A Anastasio. On hallucinations in tomographic image reconstruction. IEEE Transactions on Medical Imaging, 40(11):3249–3260, 2021.

Benjamin Boys, Mark Girolami, Jakiw Pidstrigach, Sebastian Reich, Alan Mosca, and Omer Deniz Akyildiz. Tweedie moment projected diffusions for inverse problems. Transactions on Machine Learning Research (TMLR), 2024. ISSN 2835-8856.

Benjamin A Burns and Sara Fridovich-Keil. When, why, and how do diffusion posterior samplers fail? A finite-sample lens. arXiv preprint arXiv:2605.30330, 2026.

Gabriel Cardoso, Yazid Janati El Idrissi, Sylvain Le Corff, and Eric Moulines. Monte Carlo guided denoising diffusion models for Bayesian linear inverse problems. In International Conference on Learning Representations (ICLR), volume 2024, pp. 44001–44037, 2024.

Hao Chen, Renzheng Zhang, and Scott S Howard. DAPS++: Rethinking diffusion inverse problems with decoupled posterior annealing. In European Conference on Computer Vision (ECCV), pp. 223–240. Springer, 2026.

Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 25683–25696, 2022.

Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In The Eleventh International Conference on Learning Representations (ICLR), 2023.

Evan Scope Crafts and Umberto Villa. Benchmarking diffusion annealing-based bayesian inverse problem solvers. IEEE Open Journal ofSignal Processing, 2025.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 248–255. IEEE, 2009.

Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A filtering perspective. In International Conference on Learning Representations (ICLR), volume 2024, pp. 12069–12095, 2024.

Event Horizon Telescope Collaboration. First M87 Event Horizon Telescope Results. IV. Imaging the Central Supermassive Black Hole. The Astrophysical Journal Letters, 875(1):L4, apr 2019.

James R Fienup. Phase retrieval algorithms: A comparison. Applied Optics, 21(15):2758–2769, 1982.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems (NeurIPS), volume 33, pp. 6840–6851. Curran Associates, Inc., 2020.

C¸ agatay Is¸il and Figen S Oktem. Deep plug-and-play HIO approach for phase retrieval. ˘ Applied Optics, 64(5):A84–A94, 2025.

Yazid Janati, Badr Moufad, Mehdi Abou El Qassime, Alain Oliviero Durmus, Eric Moulines, and Jimmy Olsson. A mixture-based framework for guiding diffusion models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, pp. 26830–26876. PMLR, 13–19 Jul 2025.

Pengfei Jin, Yiqi Tian, Kailong Fan, Bingjie Qi, and Quanzheng Li. Hallucination-aware diffusion sampling for inverse problems via robust prior updates. arXiv preprint arXiv:2606.02331, 2026.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4396–4405. IEEE, 2019.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 26565–26577. Curran Associates, Inc., 2022. doi: 10.52202/068431-1926.

Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 23593–23606, 2022.

Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. In International Conference on Learning Representations (ICLR), volume 2024, pp. 28027–28053, 2024.

Christopher Metzler, Phillip Schniter, Ashok Veeraraghavan, and Richard Baraniuk. prDeep: Robust phase retrieval with a flexible deep network. In International Conference on Machine Learning (ICML), pp. 3501–3510. PMLR, 2018.

Gareth O. Roberts and Richard L. Tweedie. Exponential convergence of Langevin distributions and their discrete approximations. Bernoulli, 2(4):341–363, 1996. doi: 10.2307/3318418.

Gideon Schwarz. Estimating the dimension of a model. The Annals of Statistics, 6(2):461–464, 1978. doi: 10.1214/aos/1176344136.

Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International conference on learning representations (ICLR), 2023.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Solving inverse problems in medical imaging with score-based generative models. In International Conference on Learning Representations (ICLR), 2022.

Darshan Thaker, Abhishek Goyal, and Rene Vidal. Frequency-guided posterior sampling for ´ diffusion-based image restoration. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12873–12882. IEEE, 2025.

Frederic Wang and Katherine Bouman. Sample-efficient evidence estimation of score based priors for model selection. In International Conference on Learning Representations (ICLR), volume 2026, pp. 74960–74987, 2026.

Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. In The Eleventh International Conference on Learning Representations (ICLR), 2023.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE transactions on image processing, 13(4):600– 612, 2004.

Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pp. 31372–31403. Curran Associates, Inc., 2023. doi: 10.52202/075280-1363.

Zihui Wu, Yu Sun, Yifan Chen, Bingliang Zhang, Yisong Yue, and Katherine Bouman. Principled probabilistic imaging using diffusion models as plug-and-play priors. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

Xingyu Xu and Yuejie Chi. Provably robust score-based diffusion posterior sampling for plugand-play image reconstruction. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems (NeurIPS), volume 37, pp. 36148–36184. Curran Associates, Inc., 2024. doi: 10.52202/079017-1140.

Martin Zach, Youssef Haouchat, and Michael Unser. A statistical benchmark for diffusion-posteriorsampling algorithms. In International Conference on Learning Representations (ICLR), volume 2026, pp. 108984–109021, 2026.

Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song. Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20895– 20905, 2025.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 586–595. IEEE, 2018.

Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu, Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki, Zachary E. Ross, Katherine L. Bouman, and Yisong Yue. InverseBench: Benchmarking plug-and-play diffusion priors for inverse problems in physical sciences. In International Conference on Learning Representations (ICLR), volume 2025, pp. 90912–90940, 2025.

Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1219–1229. IEEE, 2023.

## A SUPPLEMENTARY MATERIAL

## A.1 FULL ALGORITHM

Algorithm 1 Prior-Consistent DAPS (PC-DAPS)   
Require: measurement y, pretrained denoiser $D _ { \theta }$ , measurement operator ${ \mathcal { A } } ,$ measurement noise $\beta _ { y }$   
Require: noise schedule $\sigma _ { t _ { N _ { A } } } > \cdots > \sigma _ { t _ { 0 } } = 0 ,$ bag size $K ,$ Langevin steps $N ,$ , threshold $\gamma$   
1: $x _ { t _ { N _ { A } } } ^ { ( k ) } \sim \mathcal { N } ( 0 , \sigma _ { t _ { N _ { A } } } ^ { 2 } I )$ for $k = 1 , \ldots , K$   
2: pruned $ \mathtt { f a l s e }$   
3: for $i = N _ { A } , N _ { A } - 1 , \dots , 1$ do   
Clean-signal estimates and denoising distribution   
4: $\tilde { x } _ { 0 } ^ { ( k ) } \gets D _ { \theta } \Big ( x _ { t _ { i } } ^ { ( k ) } ; \sigma _ { t _ { i } } \Big )$   
5: if pruned = false then   
6: $\begin{array} { r } { q ( x _ { 0 } \mid x _ { t } )  \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { N } \Big ( x _ { 0 } ; \tilde { x } _ { 0 } ^ { ( k ) } , \mathrm { d i a g } ( \mathrm { V a r } _ { \mathcal { B } } ( \tilde { x } _ { 0 } ^ { ( k ) } ) ) \Big ) } \end{array}$   
7: else   
8: cluster $\{ \tilde { x } _ { 0 } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ into $\{ { \mathcal C } _ { m } \} _ { m = 1 } ^ { M }$   
9: construct $q ( x _ { 0 } \mid x _ { t } )$ using Equations (11) and (12)   
10: end if   
Langevin refinement   
11: if $i > . 1$ then   
12: $\hat { x } _ { 0 } ^ { ( k , 0 ) } \gets \tilde { x } _ { 0 } ^ { ( k ) }$   
13: for $j = 0 , \ldots , N - 1$ do   
14: $\hat { x } _ { 0 } ^ { ( k , j + 1 ) }$ ← Langevin step (Equation (7)) using the score in Equation (13)   
15: end for   
One-shot consensus pruning   
16: if ¬pruned and $\bar { r } < r ^ { \star }$ and clustering returns $M > 1$ clusters then   
17: compute $s ( \mathcal { C } _ { m } )$ for each cluster using Equation (9)   
18: $\mathcal { G } \gets \{ \mathcal { C } _ { m } : s ( \mathcal { C } _ { m } ) \geq ( 1 + \gamma ) \operatorname* { m a x } s ( \mathcal { C } _ { m } ) \}$   
19: randomly replace particles in clusters $\mathcal { C } _ { m } \not \in \mathcal { G }$ by copies from clusters in $\mathcal { G }$   
20: prune $\mathtt { d }  \mathtt { t r u e }$   
21: end if   
Re-noising   
22: $\boldsymbol { x } _ { t _ { i - 1 } } ^ { ( k ) } \gets \boldsymbol { \hat { x } } _ { 0 } ^ { ( k , N ) } + \sigma _ { t _ { i - 1 } } \boldsymbol { \epsilon } ^ { ( k ) } , \quad \boldsymbol { \epsilon } ^ { ( k ) } \sim \mathcal { N } ( \boldsymbol { 0 } , I )$   
23: else   
24: $\hat { x } _ { 0 } ^ { ( k ) } \gets \mathrm { O D E - S O L V E } ( x _ { t _ { 1 } } ^ { ( k ) } , \sigma _ { t _ { 1 } } \to 0 )$   
25: end if   
26: end for   
27: return $\{ \hat { x } _ { 0 } ^ { ( k ) } \} _ { k = 1 } ^ { K }$

## A.2 TOY EXPERIMENT DETAILS

Prior. The signals x are elements of $\mathbb { R } ^ { 2 }$ . We define a reference vector $u ,$ in this case the first standard basis vector $u = e _ { 1 }$ . The prior has two atoms at $\pm a u$ for scalar $a > 0 { : }$

$$
\begin{array} { r } { p ( x _ { 0 } ) = \frac { 1 } { 2 } \delta _ { - a u } + \frac { 1 } { 2 } \delta _ { a u } , \quad a = 1 . } \end{array}\tag{14}
$$

Observe that the prior denoiser is known exactly for EDM-parametrized forward model $x _ { \sigma } = x _ { 0 }$ +σϵ with $\epsilon \sim \mathcal { N } ( 0 , I )$

$$
D ( x , \sigma ) = \mathbb { E } [ x _ { 0 } \mid x _ { \sigma } = x ] = a \operatorname { t a n h } \left( a \langle x , u \rangle / \sigma ^ { 2 } \right) u .\tag{15}
$$

DAPS and PC-DAPS use this closed-form denoiser in place of a trained network, both for the probability-flow ODE and for all Tweedie estimates.

Measurement and likelihood. The measurement operator is a quadratic function in the inner product:

$$
\begin{array} { l } { \displaystyle { A ( x ) = f ( \langle x , u \rangle ) , \quad f ( z ) = \frac { c } { 2 a b } ( z - a ) ( z + a - b ) , } } \end{array}\tag{16}
$$

chosen to interpolate $f ( - a ) = c , f ( a + b ) = c$ and $f ( a ) = 0$ . Measurements follow $y = \mathcal { A } ( x _ { 0 } ) + n$ with $n \sim \mathcal N ( \bar { 0 } , \beta _ { y } ^ { 2 } )$ , so the likelihood is $p ( y \mid x _ { 0 } ) \stackrel { } { = } \mathcal N ( y ; \mathcal { A } ( x _ { 0 } ) , \beta _ { y } ^ { 2 } )$ . The observation is set to $y = c = \kappa \beta _ { y }$ . The precise values used in Figure 2 are $\beta _ { y } = 0 . 0 5 , b = \check { 0 } . 1 , \kappa = 2$ (thus $c = 0 . 1 )$ .

Exact posterior. By Bayes’ rule, the ground truth posterior $p ( x _ { 0 } \mid y )$ is accessible in closed form:

$$
p ( x _ { 0 } \mid y ) = ( 1 - \pi ) \delta _ { - a u } + \pi \delta _ { a u } , \quad \pi = \frac { p ( y \mid a u ) } { p ( y \mid - a u ) + p ( y \mid a u ) } .\tag{17}
$$

Observe that although $\mathcal { A } ( ( a + b ) u ) = \mathcal { A } ( - a u ) , ( a + b ) u$ has no prior mass, thus samples from (the neighborhood of) $( a + b )$ u are prior-inconsistent hallucinations.

## A.3 MORE DETAILS ON GAUSSIAN MIXTURE APPROXIMATION

To illustrate the role of the Gaussian mixture $q ( x _ { 0 } \mid x _ { t } )$ introduced in Section 4.2, used by PC-DAPS to approximate the denoising distribution $p ( \boldsymbol { x } _ { 0 } ~ | ~ \boldsymbol { x } _ { t } )$ , we show a side-by-side comparison of the Langevin dynamics refinement when using (i) the DAPS Gaussian approximation $\mathbf { \bar { \mathcal { N } } } ( x _ { 0 } ; \hat { x } _ { 0 } ( x _ { t } ) , r _ { t } ^ { 2 } I )$ and (ii) the PC-DAPS GMM approximation $q ( x _ { 0 } \mid x _ { t } )$ on a toy example.

In the illustrated example depicted in Figure 4, we consider a multimodal prior distribution on the plane $\mathbb { R } ^ { 2 }$ , and construct A to simply select the second component $x _ { 2 }$ of the signal vector $x .$ Four panes are shown depicting the Langevin algorithm procedure of a single (i) DAPS iteration, (ii) PC-DAPS iteration using the DAPS Gaussian approximation, (iii) PC-DAPS iteration using the PC-DAPS GMM approximation, and (iv) ground truth (denoiser and Langevin) iteration. The highprobability level sets of both the (approximated) denoiser and the induced Langevin target distribu tion are depicted in each pane.

Observe that, in both panes using the DAPS Gaussian approximation, the Langevin algorithm effectively searches for points (i) consistent with the provided measurement $y = x _ { 2 }$ and (ii) close to the initialization point in a Euclidean sense. As a result, the algorithm terminates by returning a likelihood-consistent yet posterior-inconsistent point. In contrast, the Gaussian mixture approximation is attracted to the region of the measurement’s preimage that is consistent with the prior thanks to other particles in the bag which explored and identified prior-consistent regions distant from the Langevin initialization. Additionally, observe that although the DAPS Gaussian is isotropic and thus places high probability outside the high-probability level set of the prior, the PC-DAPS approximation’s component covariances better adapt to the geometry of the prior’s support.

## A.4 ADDITIONAL IMPLEMENTATION CHOICES

There are two implementation choices that are orthogonal to the core PC-DAPS algorithm but improve reconstruction quality or computational efficiency in practice: skipping the final Langevin refinement and using adaptive ODE stepping. Their effects are summarized in Table 3 and described in detail below.

Skipping the last Langevin refinement. At the final annealing time $( t _ { 0 } = 0 )$ , the Langevin update is not performed, and the ODE endpoint $\hat { x } _ { 0 } ( x _ { t _ { 0 } } )$ is used as the output. When the noise is nearly zero, the Langevin step size becomes very small, and the added noise can slightly reduce sample quality.

Adaptive ODE steps. To compute the DAPS clean estimate $\hat { x } _ { 0 } ( x _ { t _ { i } } )$ , the probability flow ODE is integrated using Euler steps. One Euler step is used at high noise levels, where the clean estimate is less sensitive to the precise ODE integration, and this is increased to five steps at lower noise levels, where the prior becomes more discriminative and finer integration reduces discretization error near the data manifold. Since PC-DAPS utilizes a bag of multiple particles, reducing unnecessary ODE evaluations is important for controlling the additional computational cost. Adaptive ODE stepping is therefore used to improve the time efficiency of the particle-based sampler.

Descriptions of the implementation details of the particle clustering and pruning procedure used by PC-DAPS are given below.

![](images/b215ffe501f7b7f48dbf55f2362b548eca1a46df66471eb63b03ff04487ad780.jpg)  
Figure 4: Gaussian mixture approximations preserve prior-supported modes in Langevin refinement. Each pane illustrates N iterations of Langevin refinement from the same starting point $\scriptstyle { \hat { x } } _ { 0 }$ (black dot; path smoothed for readability). The same prior $p ( x _ { 0 } )$ (gray) and likelihood $p ( y \mid x _ { 0 } )$ (blue) are shown in each pane, while the approximation $q ( x _ { 0 } \mid x _ { t } )$ (teal) differs. When utilizing the Gaussian approximation, both DAPS (left) and PC-DAPS (middle-left) produce Langevin targets which are measurement-consistent yet prior-inconsistent, and the refinement ends outside the prior (orange diamond). In contrast, the PC-DAPS Gaussian mixture (middle-right) represents multiple prior-supported regions explored by particles in the bag, allowing Langevin refinement to move toward the prior-supported mode associated with the measurement (blue dot). The exact denoising distribution (right) exhibits the same multimodal structure.

Clustering technique. To assign particles to components, the corresponding clean samples $\{ \tilde { x } _ { 0 } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ described in Section 4.2 are downsampled to a coarse resolution $( \mathrm { e . g . , 3 2 \times 3 2 } )$ , centered, and projected onto the leading principal components that explain 90% of the variance. On this subspace, a Gaussian mixture model is fit with M selected via Bayesian information criterion (Schwarz, 1978). Each particle is then hard-assigned to its most likely component, giving a partition $\{ { \mathcal C } _ { m } \} _ { m = 1 } ^ { M }$ of the bag.

Pruning threshold. Pruning is applied to the post-Langevin samples only when the test in Equation (8) fires $( r ^ { \star } > \bar { r } )$ and the clustering returns more than one cluster. Each cluster is scored using the non-positive function Equation (9) with $\mathcal { S } = \{ 0 . 5 , 1 , 2 \}$ . Denoting $s ^ { \star } = \operatorname* { m a x } _ { m } s ( \mathcal { C } _ { m } )$ , all clusters $\mathcal { C } _ { m }$ satisfying $s ( \mathcal { C } _ { m } ) < \bar { ( 1 + \gamma ) } s ^ { \star }$ are pruned, while clusters satisfying $( 1 + \gamma ) s ^ { \star } \leq s ( \mathcal { C } _ { m } ) \leq s ^ { \star }$ are preserved. Each pruned particle is replaced by a copy of a surviving particle, cycling through the survivors. Observe that the cluster producing score $s ^ { \star }$ always survives. γ is calibrated from withincluster score variation: following Zhang et al. (2025), 10 calibration images per dataset are denoised via PC-DAPS (including the clustering procedure) without pruning. The relative average score difference for all particle pairs within each cluster is computed, and $\gamma$ is set to the 95th percentile of these differences.

Ablations of the main computational parameters and their comparisons are presented in the following paragraphs.

Ablation of ODE steps and annealing noise scheduling steps. PC-DAPS benefits from the same implementation improvements identified in DAPS. The effects of the number of ODE steps used to compute the clean estimate and the number of annealing noise-scheduling steps are ablated, and trends consistent with those reported for DAPS are observed: increasing the number of ODE steps generally improves perceptual quality, while denser annealing schedules are particularly beneficial for nonlinear inverse problems. The corresponding ablations are shown in Figure 5 and Figure 6.

Number of particles in a bag. Using more particles provides the selection procedure with more candidate trajectories, which improves its ability to distinguish distinct modes and reject priorinconsistent but measurement-consistent components. On phase retrieval, increasing the bag size substantially improves both PSNR and LPIPS, with the gains becoming smaller beyond K = 8–16. This improvement comes at an approximately linear increase in computational cost, since all K particles are propagated through the annealing schedule. Compared with DAPS best-of-k, PC-DAPS achieves higher reconstruction quality while using less sampling time for larger bag sizes, as shown in Figure 7.

Table 3: Comparison of the ODE schedule and final Langevin refinement. Ablated over 10 randomly selected FFHQ images using the PC-DAPS-1K configuration in phase retrieval task.
<table><tr><td rowspan="2">ODE steps</td><td rowspan="2">Final Langevin</td><td colspan="3">Mean reconstruction</td><td colspan="3">Oracle selection</td></tr><tr><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td rowspan="2">5</td><td>Included</td><td>29.50</td><td>0.787</td><td>0.168</td><td>29.63</td><td>0.789</td><td>0.167</td></tr><tr><td>Dropped</td><td>30.01</td><td>0.831</td><td>0.144</td><td>30.14</td><td>0.833</td><td>0.143</td></tr><tr><td rowspan="2">1</td><td>Included</td><td>25.02</td><td>0.692</td><td>0.274</td><td>28.43</td><td>0.780</td><td>0.188</td></tr><tr><td>Dropped</td><td>25.41</td><td>0.739</td><td>0.254</td><td>28.93</td><td>0.831</td><td>0.164</td></tr><tr><td rowspan="2">1→5</td><td>Included</td><td>28.11</td><td>0.774</td><td>0.181</td><td>28.34</td><td>0.778</td><td>0.179</td></tr><tr><td>Dropped</td><td>29.83</td><td>0.818</td><td>0.158</td><td>29.99</td><td>0.822</td><td>0.149</td></tr></table>

![](images/90fd99684addf6c4ee92ff37c43c195204f90348a68726cc687a7f902ef8ed8c.jpg)

![](images/9675f6b087b03a1fb8ca89f6c4c622be5a23c6f617b0356f4f32b67f870d5853.jpg)  
Figure 5: Effect of the number of ODE steps on PC-DAPS. Increasing the number of ODE steps generally improves perceptual quality (LPIPS) but can reduce PSNR, with the effect depending on the inverse problem.

A fair comparison on phase retrieval. Table 1 reports the best-of-k results used by the baseline methods, while PC-DAPS is evaluated using the mean over its K samples. For a direct com parison under the same evaluation protocol, Table 4 reports mean-of-k results for all methods. The baseline methods show lower mean performance because their k reconstructions can include measurement-consistent but prior-inconsistent solutions, such as color inversions, rotations, or trans lations, whereas best-of-k selects only the reconstruction closest to the ground truth.

## A.5 TIME COST

PC-DAPS keeps K particles in one sampling trajectory, while DAPS needs K separate runs to get K candidate reconstructions. If implementation overhead is ignored, both methods need about the same number of denoiser evaluations, which increases roughly linearly with K. PC-DAPS does have extra steps, including clustering, estimating mixture parameters, and a one-shot pruning step. However, these add only a small overhead compared to the cost of the ODE and Langevin updates.

The implementation choices in Appendix A.4, such as skipping the final Langevin refinement and using adaptive ODE steps at low noise, help lower the sampling cost of PC-DAPS. Figure 8 shows the measured wall-clock time and reconstruction quality. For super-resolution, PC-DAPS takes longer than DAPS because it uses multiple particles. For phase retrieval, though, the DAPS runtime includes four separate runs because it uses a best-of-4 evaluation protocol. PC-DAPS, on the other hand, creates the particle set in one run and selects internally. These results suggest that using multiple particles does not always mean a much higher wall-clock cost to achieve the same quality metric.

![](images/4c4fa4996fdc283c2a5bfa4cde62208386f5991317ceb70086d9464b4d1c437e.jpg)

![](images/1a183750ed836ca27f286e1fcf12d6d5668baec547facbf88dfe1c42945434fd.jpg)  
Figure 6: Effect of the number of annealing noise scheduling steps on PC-DAPS. Increasing the number of annealing steps has little effect on super-resolution and Gaussian deblurring, while phase retrieval shows substantial improvement with a finer annealing schedule.

![](images/3077e6ddeee8fa3ef25558bcecb9e1303c65f122c8a91f805e48605874b47f22.jpg)

![](images/4f3a74bce980e695ab481a20398ac6856ccaafe72b3412b17dd8ecfd6a1e79e2.jpg)  
Figure 7: Effect of the number of particles in a bag on phase retrieval. Increasing the bag size improves PC-DAPS reconstruction quality, with both PSNR and LPIPS improving rapidly up to $K = 8 – 1 6$ and then beginning to saturate. PC-DAPS is competitive with DAPS best-of-k at comparable values of k, while requiring less sampling time for larger k.

## A.6 INVERSE PROBLEM SETUP

In Table 1, following the inverse-problem settings of DAPS (Zhang et al., 2025), the benchmark includes the eight tasks from DAPS: 4× super-resolution, box inpainting, random inpainting, Gaussian deblurring, motion deblurring, nonlinear deblurring, phase retrieval, and high dynamic range (HDR) reconstruction. For these tasks, the same measurement operators, measurement noise level, and evaluation protocol as DAPS are used. Unless otherwise stated, measurements are corrupted with additive Gaussian noise with standard deviation $\beta _ { y } = 0 . 0 5$ . Methods are evaluated on 100 test images from each dataset, FFHQ and ImageNet, at a resolution of 256×256. Additionally, 16× super-resolution and high-pass filtering from Thaker et al. (2025) are included in the table and described in detail below.

Super-resolution 16×. Bicubic downsampling is applied from 256×256 to $1 6 \times 1 6 ,$ followed by additive Gaussian noise with standard deviation $\beta _ { y } ~ = ~ 0 . 0 5$ This uses the same linear bicubic downsampling operator as the 4× super-resolution task in DAPS, but with a larger downsampling factor. The resulting operator has a larger null space, leaving more image content unconstrained by the measurement and providing a more challenging setting. All other sampler settings follow the 4× super-resolution configuration.

Table 4: Phase retrieval under a mean-of-k evaluation. All methods are evaluated using the mean performance across k reconstructions, rather than selecting the reconstruction closest to the ground truth. This exposes failures that best-of-k evaluation can mask, since individual reconstructions may be measurement-consistent but differ from the ground truth due to color inversion, rotation, or translation. PC-DAPS achieves higher reconstruction quality compared to other baselines.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="3">FFHQ</td><td colspan="3">ImageNet</td></tr><tr><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td></tr><tr><td rowspan="6">Phase retrieval</td><td>PC-DAPS</td><td> ${ \bf 3 0 . 0 9 } _ { \pm 1 . 7 7 }$ </td><td> $\mathbf { 0 . 8 3 4 _ { \pm 0 . 0 8 9 } }$ </td><td> $\mathbf { 0 . 1 5 2 } _ { \pm 0 . 0 8 8 }$ </td><td> $2 1 . 7 5 _ { \pm 6 . 1 0 }$ </td><td> $\mathbf { 0 . 4 6 9 _ { \pm 0 . 1 3 2 } }$ </td><td> $\mathbf { 0 . 3 7 3 _ { \pm 0 . 1 2 8 } }$ </td></tr><tr><td>DAPS</td><td> $2 5 . 1 1 _ { \pm 5 . 7 3 }$ </td><td> $0 . 6 6 6 _ { \pm 0 . 1 4 1 }$ </td><td> $0 . 2 8 3 _ { \pm 0 . 1 3 8 }$ </td><td> $1 5 . 7 0 { \scriptstyle \pm 3 . 9 0 }$ </td><td> $0 . 3 1 7 _ { \pm 0 . 1 5 3 }$ </td><td> $0 . 5 1 6 _ { \pm 0 . 1 0 1 }$ </td></tr><tr><td>DPS</td><td> $1 4 . 1 3 { \scriptstyle \pm 2 . 7 2 }$ </td><td> $0 . 3 9 8 { \scriptstyle \pm 0 . 1 0 6 }$ </td><td> $0 . 5 0 8 { \scriptstyle \pm 0 . 0 8 5 }$ </td><td> $1 3 . 3 8 { \scriptstyle \pm 4 . 0 3 }$ </td><td> $0 . 3 1 1 { \scriptstyle \pm 0 . 2 0 8 }$ </td><td> $0 . 5 6 0 { \scriptstyle \pm 0 . 1 2 3 }$ </td></tr><tr><td>PnP-DM</td><td> $1 7 . 6 8 { \scriptstyle \pm 5 . 5 0 }$ </td><td> $0 . 4 5 8 { \scriptstyle \pm 0 . 1 9 1 }$ </td><td> $0 . 5 2 3 { \scriptstyle \pm 0 . 1 4 9 }$ </td><td> $1 4 . 8 0 { \scriptstyle \pm 5 . 6 0 }$ </td><td> $0 . 3 8 2 _ { \pm 0 . 1 7 0 }$ </td><td> $0 . 5 7 0 { \scriptstyle \pm 0 . 1 1 0 }$ </td></tr><tr><td>DPnP</td><td> $1 6 . 5 7 _ { \pm 6 . 7 6 }$ </td><td> $0 . 4 7 9 _ { \pm 0 . 2 1 3 }$ </td><td> $0 . 4 9 9 _ { \pm 0 . 1 9 3 }$ </td><td> $1 4 . 6 9 _ { \pm 3 . 6 1 }$ </td><td> $0 . 4 2 7 _ { \pm 0 . 1 8 0 }$ </td><td> $0 . 5 8 4 _ { \pm 0 . 1 0 1 }$ </td></tr><tr><td>RED-Diff</td><td> $1 4 . 3 0 _ { \pm 3 . 9 0 }$ </td><td> $0 . 3 4 7 _ { \pm 0 . 1 8 0 }$ </td><td> $0 . 6 3 0 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td> $1 3 . 9 0 _ { \pm 3 . 3 0 }$ </td><td> $0 . 3 5 4 { \scriptstyle \pm 0 . 0 7 0 }$ </td><td> $0 . 5 5 8 { \scriptstyle \pm 0 . 1 0 5 }$ </td></tr></table>

![](images/f8c767c2ff6662563f037f5070b9c5bdcf1a7b5b12b66c33534cc1e6d17c6f44.jpg)

![](images/aa68b9bc9ec5dc472fb41f2af3c478e98817b8d737a40f3ccdbbfb945414cd0c.jpg)  
(a) 4 × super-resolution  
(b) Phase retrieval  
Figure 8: Comparison of single input sampling time and reconstruction quality between DAPS and PC-DAPS. The x-axis shows the sampling time for a single input sample measured on an NVIDIA A100-SXM4-80GB GPU, while the y-axis reports LPIPS. Evaluation is performed on 100 FFHQ images. For phase retrieval, DAPS requires four independent runs for an input image; therefore, the reported DAPS sampling time corresponds to the total time of all four runs.

High-pass filtering. Following Thaker et al. (2025), a 61×61 convolutional kernel is used, constructed as a Dirac delta minus a Gaussian blur kernel with standard deviation 5.0. The resulting kernel has zero sum and therefore removes the DC component. Gaussian noise with standard deviation $\beta _ { y } = 0 . 0 5$ is added to the filtered image.

## A.6.1 BASELINE DETAILS

For DAPS (Zhang et al., 2025), DPS (Chung et al., 2023), DDNM (Wang et al., 2023), FPS-SMC (Dou & Song, 2024), RED-Diff (Mardani et al., 2024), FGPS (Thaker et al., 2025), PnP-DM (Wu et al., 2024), and DPnP (Xu & Chi, 2024), the authors’ official public implementations and recommended task-specific hyperparameters are used throughout the benchmark. DAPS uses the DAPS-4K configuration for nonlinear tasks and DAPS-1K for linear tasks. For consistency in computational budget, PC-DAPS uses the DAPS-1K configuration for both linear and nonlinear tasks, including its adaptive variant across all tasks. When a baseline specifies task-specific step sizes or noise-schedule parameters, the reported values are used without modification. HIO, a classical phase retrieval method, is included as an additional reference baseline, with results taken from Zhang et al. (2025).

For the two additional tasks not included in the DAPS benchmark, 16× super-resolution and highpass filtering, the closest corresponding configuration available for each baseline is used: the 4× super-resolution configuration for 16× super-resolution and the Gaussian deblurring configuration for high-pass filtering. Only the measurement operator and the associated measurement generation are modified to match the task definitions described above.

Unless otherwise specified, PC-DAPS uses the 1K setting with 200 annealing steps and adaptive ODE stepping, and generates K = 4 reconstructions for each measurement. Reported PC-DAPS metrics are averaged over the four reconstructions. For phase retrieval, baselines other than PC-

Table 5: Sampling configurations. ODE steps, annealing steps and number of function evaluations (NFE) for the DAPS and PC-DAPS configurations used in the ablation studies. PC-DAPS uses a bag of K = 4 particles; 1→k means one Euler step for the first 65% of the annealing steps and k steps afterwards. NFE is the number of denoiser evaluations per sample (per run for DAPS, per particle for PC-DAPS), counted during sampling. PC-DAPS configurations are named after the DAPS configuration with the same annealing steps and final ODE steps; they use fewer ODE steps early on but one extra denoiser evaluation per annealing step to fit the mixture, so their NFE differs from the name.
<table><tr><td>Configuration</td><td>DAPS-50</td><td>DAPS-100</td><td>DAPS-200</td><td>DAPS-400</td><td>DAPS-1K</td><td>DAPS-2K</td><td>DAPS-4K</td></tr><tr><td>ODE Steps</td><td>2</td><td>2</td><td>2</td><td>4</td><td>5</td><td>8</td><td>10</td></tr><tr><td>Annealing Steps</td><td>25</td><td>50</td><td>100</td><td>100</td><td>200</td><td>250</td><td>400</td></tr><tr><td>NFE</td><td>50</td><td>100</td><td>200</td><td>400</td><td>1000</td><td>2000</td><td>4000</td></tr><tr><td>Configuration</td><td>PC-DAPS-50</td><td>PC-DAPS-100</td><td>PC-DAPS-200</td><td>PC-DAPS-400</td><td>PC-DAPS-1K</td><td></td><td></td></tr><tr><td>ODE Steps</td><td>1→2</td><td>1→2</td><td>1→2</td><td>1→4</td><td>1→5</td><td></td><td></td></tr><tr><td>Annealing Steps</td><td>25</td><td>50</td><td>100</td><td>100</td><td>200</td><td></td><td></td></tr><tr><td>NFE</td><td>66</td><td>131</td><td>261</td><td>331</td><td>732</td><td></td><td></td></tr></table>

DAPS are evaluated using the best of four independent runs, selecting the reconstruction closest to the ground-truth image, following prior work (Chung et al., 2023; Zhang et al., 2025). This protocol gives the baselines access to ground-truth information that is unavailable to PC-DAPS at test time. PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018) are reported in Table 1 for all methods. For ablation studies, the PC-DAPS variants follow the corresponding DAPS settings reported in Table 5.

Sample 4

Sample 4

Sample 2

Sample 3

Sample 3

Sample 1

Sample 1

Reference

## A.7 MORE QUALITATIVE SAMPLES

Sample 2  
![](images/1295d82cf64152bee636c0ce9acfb84d738ee11f2a1d84c955ab4fba6a275a6b.jpg)  
Figure 9: PC-DAPS samples for phase retrieval on FFHQ. Samples are shown without manual selection to better visualize success rate and sample quality.

Reference

Sample 3

Sample 1  
Sample 2  
Sample 3  
Sample 4  
Reference  
Sample 1  
Sample 2  
Sample 4  
![](images/36a2106da39119fec9292be81e4a5b8e58811699d31f97af07adca7924d2154a.jpg)  
Figure 10: PC-DAPS samples for phase retrieval on ImageNet. Samples are shown without manual selection to better visualize the success rate and sample quality. As discussed in Section 5, ImageNet samples in a bag still contain failed (diverged/noisy) particles and underperform compared to FFHQ.

Measurement  
Reference  
Sample 1  
Sample 2  
Sample 3  
Sample 4  
![](images/09e419254ea3a90acc92d3f65c02d77442c7cd4bf00306bc0916e56f9c494c4a.jpg)  
Figure 11: PC-DAPS samples for 16× super-resolution. PC-DAPS is able to keep the diversity in the multimodal posterior distribution.

Measurement  
Reference  
Sample 1  
Sample 2  
Sample 3  
Sample 4  
![](images/e1e8ab7fae50135366932ae8e57d4e2cf915c713ae58ecd724c8904e166bba48.jpg)  
Figure 12: PC-DAPS samples for 190×190 box inpainting. PC-DAPS is able to keep the diversity in the multimodal posterior distribution.

![](images/b658ddb69e4840402f6bec6e0900a067e271f8e293b88bd43cfa8c2f812d027a.jpg)  
Figure 13: PC-DAPS samples for various tasks on FFHQ. PC-DAPS obtains competitive reconstructions for linear and nonlinear tasks.

Reference  
Sample 1  
Sample 2  
Measurement  
Reference  
Sample 1  
Sample 2  
![](images/9d3926cfe905d9d06b8490b80e6dec25641fdc3bd5d5cbca80accad80471bedd.jpg)  
Figure 14: PC-DAPS samples for various tasks on ImageNet. PC-DAPS obtains competitive reconstructions for linear and nonlinear tasks.