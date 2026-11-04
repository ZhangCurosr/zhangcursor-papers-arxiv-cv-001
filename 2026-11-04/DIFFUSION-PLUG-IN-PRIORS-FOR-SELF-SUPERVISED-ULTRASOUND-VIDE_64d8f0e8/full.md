# DIFFUSION PLUG-IN PRIORS FOR SELF-SUPERVISED ULTRASOUND VIDEO RESTORATION

Chu Chen<sup>1,∗</sup> Mingduo He<sup>2,∗</sup> Yining Li<sup>3,∗</sup> Yinhao Wu<sup>4</sup> Wei Tang<sup>5,6</sup> Yiwen Luo<sup>7</sup> Yixuan Chen<sup>5</sup> Graham Treece<sup>1</sup> Pasquale Cascarano<sup>8</sup> Kangning Cui<sup>9,10,†</sup>

<sup>1</sup>Department of Engineering, University of Cambridge, Cambridge, UK <sup>2</sup>Department of Biomedical Engineering, School of Life Science and Technology, Huazhong University of Science and Technology, Wuhan, China <sup>3</sup>Department of Intelligent Medical Engineering, School of Medicine, Nankai University, Tianjin, China <sup>4</sup>University of Texas at Arlington, Arlington, TX, USA <sup>5</sup>Department of Engineering Science, University of Oxford, Oxford, UK <sup>6</sup>Oxford Suzhou Centre for Advanced Research, Suzhou, China <sup>7</sup>Department of Electrical Engineering, City University of Hong Kong, Hong Kong SAR <sup>8</sup>Department of the Arts, University of Bologna, Bologna, Italy <sup>9</sup>Department of Data Science, City University of Hong Kong (Dongguan), Dongguan, China <sup>10</sup>City University of Hong Kong, Hong Kong SAR

<sup>∗</sup>Equal contribution, listed in alphabetical order. <sup>†</sup>Corresponding author: kangning.cui@cityu-dg.edu.cn

## ABSTRACT

Objective: Ultrasound videos are limited by hardware-constrained spatial resolution and signal dependent speckle, while paired low- and high-resolution acquisitions for supervised restoration are rarely available. We aim to jointly super-resolve and despeckle ultrasound videos without paired training data or network retraining. Methods: We propose UltraDiff, which uses a frozen pretrained diffusion model (DM) as a plug-in prior and restores each frame by optimizing the latent seed of the DM sampler under a range-constrained maximum a posteriori formulation. We derive a speckle-aware data-fidelity term from a Rayleigh scale-family model and characterize its relativeresidual and variance-adaptive weighting properties. We further introduce a stabilized fidelity that is bounded below and strictly convex in log intensity for positive observations in its unclamped form. A total-variation regularizer complements the DM prior, and a temporal seed update warmstarts each frame from the optimized seed of its predecessor. Results: On EchoNet-LVH videos with simulated speckle and additive noise, UltraDiff with a natural-image prior improves PSNR by 1.24 dB over the strongest baseline, and ultrasound-related priors improve it further. On real clinical echocardiography videos, UltraDiff reduces the mean absolute error of zero-shot ejection fraction (EF) prediction from 7.31 to 6.37 EF points relative to DUP, the strongest baseline. Conclusion: Combining a speckle-aware fidelity with frozen diffusion priors enables ground-truth-free ultrasound video restoration that benefits downstream measurement. Significance: UltraDiff needs neither paired data nor retraining and is not tied to a specific diffusion checkpoint. Code associated with this work will be made publicly available upon acceptance.

Keywords ultrasound · video super-resolution · diffusion model · self-supervised learning · despeckling · inverse problems

## 1 Introduction

Ultrasound (US) imaging is widely used in cardiology [1–3], obstetrics [4,5], and musculoskeletal imaging [6] because it is real-time, non-invasive, and cost-effective. Compared with static US images, US videos capture dynamic tissue motion, which is essential for applications such as breast lesion assessment [7,8] and cardiovascular monitoring [9–11]. In echocardiography, ejection fraction (EF) estimation depends on clear cardiac structures throughout the cardiac cycle.

![](images/ae138937b2e6dc91c83ddc2e4fb2ad97181f4208227c54a63bba548c1407f30d.jpg)  
Figure 1: Motivating comparison of ultrasound video restoration. We compare reconstructed frames from LR ultrasound sequences with HR references and representative methods, including EDSR, ZSSR-Cycle, and UltraDiff. The zoom-in regions highlight structural details and tissue texture: UltraDiff suppresses noise while better preserving anatomical structures and the tissue echotexture.

Despite these advantages, the quality of US videos is inherently constrained by the physics of acoustic imaging. Spatial resolution is limited by the frequency-penetration trade-off of transducer design, while image quality is further affected by signal-dependent speckle noise caused by the interference of backscattered waves [12]. Classical despeckling methods reduce these fluctuations [12,13], but must balance noise suppression against the loss of tissue echotexture, which itself carries diagnostic information. These degradations reduce boundary visibility and can compromise downstream measurements such as automated EF prediction. Super-resolution (SR) offers a computational route beyond hardwarelimited resolution [14]. However, robust US video restoration remains challenging: paired low-resolution (LR) and high-resolution (HR) US videos are rarely available, acquisition conditions vary across devices and operators, and US noise statistics differ substantially from natural-image degradations. As illustrated in Fig. 1, representative SR baselines either over-smooth cardiac structures or fail to recover realistic tissue texture.

## 1.1 Related Work and Motivation

SR methods for natural images and videos have advanced rapidly with large-scale paired datasets [15–17]. Singleimage SR methods commonly use deep convolutional networks [18,19], while video SR methods incorporate temporal information through flow-based [20] or attention-based mechanisms [21]. However, these models are typically trained on fixed degradation distributions and may generalize poorly to clinical US videos, where image appearance, resolu tion, and speckle patterns vary across acquisition settings.

In US image SR, paired LR and HR data of the same anatomy are particularly difficult to obtain, so existing studie rely on simulated pairs or self-supervised strategies. Choi et al. [22] adapted SRGAN [23] for B-mode US image enhancement, while a U-Net-style architecture [24] was trained on simulated pairs for super-resolved US reconstruction. Liu et al. [25] proposed ZSSR-Cycle, a zero-shot and self-supervised generative adversarial framework for US image SR. A related line of work exploits the implicit prior of network architectures: Deep Image Prior (DIP) [26] fits a randomly initialized network to a single degraded image, and its video extensions propagate network weights or priors across frames [27, 28]. Following this paradigm, DUP [29] performs video-adaptive, ground-truth-free US video quality improvement. These methods re-fit network parameters for every input and rely on early stopping to avoid fitting noise. Moreover, they do not explicitly couple a learned generative prior with a speckle-aware US measurement model.

![](images/be53712e9fa95cf67bbbb82c16bd9a40dfba3ffd496c351e7c6b885cecd62cc9.jpg)  
Figure 2: Overview of the UltraDiff framework. A frozen pre-trained diffusion model maps latent seeds to HR estimates, which are degraded back to the LR space and optimized against the observed LR frames. The optimized seeds are propagated and perturbed across consecutive frames to enable video-adaptive restoration without diffusion model retraining.

Diffusion models (DMs) have been widely used as generative priors for image restoration, improving perceptual detail through iterative denoising [30, 31]. Diffusion-based methods have been explored for video enhancement [32] and US image SR [33]. DMPlug [34] shows that a pre-trained DM can be used as a plug-in prior for inverse problems without task-specific retraining, by optimizing the input of the sampler under a least-squares fidelity on single images. Applying such priors to US video restoration, however, is non-trivial: (i) a least-squares fidelity implicitly assumes signal-independent noise, which is inconsistent with multiplicative speckle; (ii) the method must adapt to each LR video without paired HR supervision; and (iii) temporal redundancy should be exploited without training additional temporal modules.

## 1.2 Contributions

To address these challenges, we propose UltraDiff, a self-supervised framework for ultrasound video super-resolution and despeckling. UltraDiff uses a frozen pre-trained diffusion model as a plug-in prior and adapts it to each LR video by optimizing latent seeds rather than model parameters. The optimized HR frames are constrained by a degradation operator and a speckle-aware data-fidelity term, so restoration is guided by both the learned image prior and the ultrasound measurement model. The main contributions of this work are summarized as follows:

• We propose UltraDiff, a self-supervised US video restoration framework for joint super-resolution and despeckling that requires neither paired HR videos nor DM retraining.

• We develop a speckle-aware restoration objective based on a Rayleigh scale-family model and a rangeconstrained MAP-inspired formulation. We characterize its relative-residual behavior and local varianceadaptive weighting, and introduce a stabilized fidelity with favorable optimization properties.

• We introduce a temporal seed update (SUP) strategy that warm-starts each frame from the optimized seed of its predecessor, combining the previous optimized seed with Gaussian perturbations and reducing the iteration budget for subsequent frames.

• We validate UltraDiffon two public echocardiography datasets, showing improved restoration quality, noise robustness, and zero-shot EF prediction over representative SR baselines, with ablations verifying key components.

## 2 Methodology

## 2.1 Problem Formulation

We consider frame-wise restoration of an LR US video $\{ I _ { t } ^ { L R } \} _ { t = 1 } ^ { T }$ under the observation model [12]

$$
I _ { t } ^ { L R } = ( A I _ { t } ^ { H R } ) \odot \phi _ { t } + \psi _ { t } , \quad t = 1 , 2 , \ldots , T ,\tag{1}
$$

where ⊙ denotes element-wise multiplication, $\mathcal { A } \in \mathbb { R } ^ { M N \times s ^ { 2 } M N }$ is a linear downsampling operator, $I _ { t } ^ { L R } \in \mathbb { R } ^ { M N }$ and $I _ { t } ^ { H R } \in \mathbb { R } ^ { s ^ { 2 } M N }$ are the vectorized t-th LR and HR frames with intensities normalized to [0, 1], s is the upscaling factor, and $T$ is the number of frames. The terms $\phi _ { t } , \psi _ { t } \in \mathbb { R } ^ { M N }$ denote multiplicative speckle and additive noise, respectively, with i.i.d. entries, mutually independent and independent of $I _ { t } ^ { H R } , \dot { \mathbb { E } } [ \phi _ { t } ( i ) ] = 1$ , and $\mathbb { E } [ \psi _ { t } ( i ) ] = 0$ . For brevity, we write $y _ { i } : = I _ { t } ^ { L R } ( i )$ and $x _ { i } \stackrel { \cdot } { : } = [ \dot { A I } _ { t } ^ { H R } ] ,$ <sub>i</sub> and drop the frame index when no ambiguity arises. Under (1), $\mathbb { E } [ y _ { i } \mid \dot { x } _ { i } ] = x _ { i }$ and

$$
\mathrm { V a r } [ y _ { i } \mid x _ { i } ] = \sigma _ { \phi } ^ { 2 } x _ { i } ^ { 2 } + \sigma _ { \psi } ^ { 2 } ,\tag{2}
$$

where $\sigma _ { \phi } ^ { 2 }$ and $\sigma _ { \psi } ^ { 2 }$ are the variances of $\phi _ { t } ( i )$ and $\psi _ { t } ( i )$ . The first term is signal-dependent, while the second term dominates where $x _ { i }$ is small, e.g., in anechoic cavities and outside the imaging sector.

We use (1) as a practical observation model for normalized B-mode video frames, which combines signal-dependent speckle with additive perturbations [12, 13]. The multiplicative component motivates a scale-family likelihood $p ( y _ { i }$ $x _ { i } ) = x _ { i } ^ { - 1 } f ( y _ { i } / x _ { i } )$ , which provides the basis for our speckle-aware data fidelity. The diffusion generator and TV regularization further constrain the reconstruction under mixed-noise degradation (Section 2.3).

## 2.2 Speckle-Aware Data Fidelity

To construct a tractable speckle-aware fidelity, we start from the Rayleigh model of fully developed envelope speckle, whose distribution scale is proportional to local reflectivity [12]:

$$
p ( y _ { i } \mid x _ { i } ) = { \frac { y _ { i } } { \beta ^ { 2 } x _ { i } ^ { 2 } } } \exp \left( - { \frac { y _ { i } ^ { 2 } } { 2 \beta ^ { 2 } x _ { i } ^ { 2 } } } \right) , \quad y _ { i } \geq 0 ,\tag{3}
$$

where $\beta > 0$ links the Rayleigh scale to $x _ { i } ;$ since $\mathbb { E } [ y _ { i } \mid x _ { i } ] = \beta { \sqrt { \pi / 2 } } x _ { i } .$ , the choice $\beta = \sqrt { 2 / \pi }$ yields unit-mean speckle. Eq. (3) has the scale-family form with $f ( r ) = ( r / \beta ^ { 2 } ) \dot { \exp } ( - r ^ { 2 } / ( 2 \beta ^ { 2 } ) )$ . The negative log-likelihood of a frame is

$$
- \log p ( I _ { t } ^ { L R } \mid I _ { t } ^ { H R } ) = \sum _ { i = 1 } ^ { M N } \ell ( x _ { i } ; y _ { i } ) + \sum _ { i = 1 } ^ { M N } \bigl ( 2 \log \beta - \log y _ { i } \bigr ) ,\tag{4}
$$

where the second sum does not depend on $I _ { t } ^ { H R }$ and the per-pixel term is $\ell ( x ; y ) : = 2 \log x + y ^ { 2 } / ( 2 \beta ^ { 2 } x ^ { 2 } )$ . Its structure is characterized as follows (proofs are given in the Appendix).

Proposition 1 Let $y > 0$ and $x > 0 .$

(i) Log-residualform: with $r = \log y - \log x$

$$
\ell ( x ; y ) = 2 \log y + \rho ( r ) , \quad \rho ( r ) = \frac { e ^ { 2 r } } { 2 \beta ^ { 2 } } - 2 r ,\tag{5}
$$

where $\rho$ is strictly convex. Hence ℓ has the unique minimizer $\bar { x } = y / ( \sqrt { 2 } \beta )$ and is scale-equivariant, i.e., $\ell ( c x ; c y ) =$ $\ell ( x ; y ) + 2$ log c for all $c > 0$

(ii) Local weighting: around x¯,

$$
\ell ( x ; y ) = \ell ( \bar { x } ; y ) + \frac { 2 } { \bar { x } ^ { 2 } } ( x - \bar { x } ) ^ { 2 } + O ( | x - \bar { x } | ^ { 3 } ) .\tag{6}
$$

Property (i) shows that ℓ penalizes the log-ratio between observation and prediction rather than their difference, so a given absolute error costs less in a bright myocardial region than in a dark one, in agreement with the proportionally larger fluctuations of bright regions in (2). Property (ii) makes this precise: near its minimizer, ℓ is a weighted least-squares term with weight $\propto 1 / \hat { x } ^ { 2 }$ , i.e., inversely proportional to the speckle variance $\sigma _ { \phi } ^ { 2 } x ^ { 2 }$ . By contrast, the common mean squared error (MSE) (e.g., used in DMPlug [34]) has constant curvature and implicitly assumes signalindependent noise, so it over-weights bright, high-variance regions and encourages fitting speckle there. In the log intensity domain, ℓ is an additive penalty $\rho$ on the log-residual, i.e., the negative log-likelihood of log-Rayleigh noise. Proposition 1 characterizes the per-pixel geometry only. The overall objective remains nonconvex in the latent seed.

Two issues prevent using ℓ directly. First, zero-valued observations commonly occur in masked background regions and anechoic cavities. For such pixels, the reduced fidelity $\ell ( x ; 0 ) = 2 \log \bar { x }$ decreases without bound as $x  0 .$ causing excessive influence from background pixels during optimization. Second, the weight $2 / \bar { x } ^ { 2 }$ in (6) diverges in dark regions, where by (2) the additive term dominates, and the speckle likelihood is misspecified. We therefore use the stabilized fidelity

$$
\mathcal { L } _ { \mathrm { d a t a } } ( I _ { t } ^ { L R } , I ) = \frac { 1 } { M N } \sum _ { i = 1 } ^ { M N } \ell _ { s } \big ( [ A I ] _ { i } ; y _ { i } \big ) ,\tag{7}
$$

with $\ell _ { s } ( x ; y ) = 2 \log ( 1 + x ) + y ^ { 2 } / ( 2 \beta ^ { 2 } \operatorname* { m a x } ( x , \epsilon ) ^ { 2 } )$ , where the clamp at a small $\epsilon > 0$ avoids division by zero.

Proposition 2 For $x > 0$ and $y \geq 0$ (ignoring the clamp):

$( i ) \ell _ { s } ( x ; y ) \ge 0 ,$ , and the gradient ofits logarithmic term is bounded by 2.

(ii) For $y > 0 , \ell _ { s }$ is strictly convex in u = log x and has a unique stationary point $x ^ { * }$ , the positive root of $2 x ^ { 3 } =$ $\beta ^ { - 2 } y ^ { 2 } ( 1 + x ) ; f o r \beta = 1$ and $y \in ( 0 , 1 ] , y \leq x ^ { \ast } \leq 1$ , with $x ^ { * } = y$ ifand only $i f y = 1$

$( i i i ) \ A t \ x ^ { * } , \ \partial _ { x } ^ { 2 } \ell _ { s } = ( 6 + 4 x ^ { * } ) / \big ( x ^ { * } ( 1 + x ^ { * } ) ^ { 2 } \big )$ , which decreases monotonically in $x ^ { * }$ and grows as $O ( 1 / x ^ { * } )$ when $x ^ { * } \to 0 ,$ compared with $4 / \bar { x } ^ { 2 }$ for ℓ and a constant for the MSE.

Thus, $\ell _ { s }$ removes the degeneracy of ℓ while retaining its variance-adaptive behavior: bright pixels are still downweighted relative to dark ones, but the dark-region weight grows as $O ( 1 / x )$ instead of $O ( 1 / x ^ { 2 } )$ , interpolating between multiplicative-noise and homoscedastic weighting in line with the mixed structure of (2). For $\beta = 1$ and $y \in ( 0 , 1 ]$ the per-pixel stationary point lies in $[ y , 1 ]$ , reflecting the intensity-dependent shift introduced by stabilization. The estimate, however, is not determined pixel-wise: A couples neighboring HR pixels, the HR frame is restricted to the generator (Section 2.3), and the optimization is early-stopped, so the reconstruction is governed jointly by fidelity, prior, and regularizer (accuracy is assessed in Section 3). In the stabilized fidelity, we set $\beta = 1$ to fix the relative scaling between the logarithmic and signal-dependent terms throughout all experiments.

## 2.3 Diffusion Plug-in Prior and Overall Objective

Fig. 2 summarizes the framework. Let $\mathcal { M } : \mathbb { R } ^ { d }  \mathbb { R } ^ { s ^ { 2 } M N }$ denote the deterministic K-step DDIM sampler [30] of a frozen pre-trained DM, which maps an initial latent seed z to an image, followed by an affine rescaling to [0, 1]. Following DMPlug [34], we treat M as a fixed generator and parameterize the HR frame as $I _ { t } ^ { H R } = \mathcal { M } ( z )$ Combining the stabilized data fidelity with a prior that restricts the HR frame to the range $\mathcal { R } ( \mathcal { M } ) = \{ \mathcal { M } ( z ) : z \in \mathbb { R } ^ { d } \}$ and penalizes its total variation (TV), we obtain the range-constrained MAP-inspired optimization problem

$$
\hat { I } _ { t } = \mathop { \mathrm { a r g m i n } } _ { I \in \mathcal { R } ( \mathcal { M } ) } \mathcal { L } _ { \mathrm { d a t a } } ( I _ { t } ^ { L R } , I ) + \lambda R ( I ) ,\tag{8}
$$

which is solved in the latent space as

$$
z _ { t } ^ { * } = \underset { z } { \mathrm { a r g m i n } } \ \mathcal { L } _ { \mathrm { d a t a } } ( I _ { t } ^ { L R } , \mathcal { M } ( z ) ) + \lambda R ( \mathcal { M } ( z ) ) ,\tag{9}
$$

with $\hat { I } _ { t } = \mathcal { M } ( z _ { t } ^ { * } )$ . Here, λ controls the regularization strength and R is the anisotropic TV [35]

$$
R ( I ) = \frac { 1 } { s ^ { 2 } M N } \sum _ { i = 1 } ^ { s ^ { 2 } M N } \big ( | ( \Delta _ { h } I ) _ { i } | + | ( \Delta _ { v } I ) _ { i } | \big ) ,\tag{10}
$$

where $\Delta _ { h }$ and $\Delta _ { v }$ are first-order horizontal and vertical finite-difference operators. Both terms in (9) are normalized per pixel, so λ is comparable across resolutions. The DM weights remain fixed, and $\nabla _ { z }$ is obtained by backpropagation through the K sampler steps.

## 2.4 Temporal Seed Update

For a video, (9) is solved frame by frame. Rather than initializing each seed independently $( z _ { t } = n _ { t } , n _ { t } \sim \mathcal { N } ( 0 , I ) )$ which discards temporal context, or reusing a fixed seed, which limits frame-specific adaptation, the seed update (SUP) strategy initializes

$$
z _ { t } ^ { ( 0 ) } = { \binom { n _ { 1 } , \quad } { z _ { t - 1 } ^ { * } + \sigma n _ { t } , } } \in \mathbb { 1 } ,\tag{11}
$$

where σ controls the perturbation magnitude. We offer three remarks on this design.

Algorithm 1 UltraDiff for an LR ultrasound video   
Require: $\{ I _ { t } ^ { L R } \} _ { t = 1 } ^ { T } , \mathcal { M } , \mathcal { A } , \lambda , \sigma , \eta , \epsilon , ( K _ { \operatorname* { m i n } } ^ { ( t ) } , K _ { \operatorname* { m a x } } ^ { ( t ) } ) , P , \epsilon _ { s }$   
Ensure: HR frames $\{ \hat { I } _ { t } \} _ { t = 1 } ^ { T }$   
1: for t = 1 to T do   
2: Initialize z by (11); $\mathcal { L } _ { \mathrm { b e s t } }  \infty ; c  0$   
3: for $k = 1$ to $K _ { \underline { { \mathrm { m a x } } } } ^ { ( t ) }$ do   
4: $\mathcal { L } \gets \mathcal { L } _ { \mathrm { d a t a } } ( I _ { t } ^ { L R } , \mathcal { M } ( z ) ) + \lambda R ( \mathcal { M } ( z ) )$   
5: $\boldsymbol { z } \gets \mathrm { A d a m } ( \boldsymbol { z } , \nabla _ { \boldsymbol { z } } \mathcal { L } , \eta )$   
6: if $\mathcal { L } < ( 1 - \dot { \epsilon } _ { s } ) \mathcal { L } _ { \mathrm { b e s t } }$ then   
7: $\mathcal { L } _ { \mathrm { b e s t } }  \mathcal { L } ; c  0$   
8: else if $k > K _ { \operatorname* { m i n } } ^ { ( t ) }$ then   
9: $c  c + 1 ; \mathbf { \overrightarrow { i f } } c \geq P$ then break   
10: end if   
11: end for   
12: $z _ { t } ^ { * } \gets z ; \hat { I } _ { t } \gets \mathcal { M } ( z _ { t } ^ { * } )$   
13: end for

Perturbation strength. With $\sigma = 0 ;$ , the previous optimized seed is reused without fresh exploration, while as σ grows, fresh noise contributes more strongly to the initialization. SUP with small σ balances information reuse and frame-specific adaptation.

Bounded initial discrepancy. Since M is a composition of K evaluations of a smooth network, it is locally Lipschitz: with a local constant L around $z _ { t - 1 } ^ { * } , \| \mathcal { M } ( z _ { t } ^ { ( 0 ) } ) - \hat { I } _ { t - 1 } \| _ { 2 } \leq L \sigma \| n _ { t } \| _ { 2 }$ . Although L is unknown and possibly large, the initial discrepancy thus scales with σ, in line with the observation that nearby DDIM latents map to similar images [30]. Because consecutive echocardiographic frames differ only slightly at typical frame rates [36], this warm start motivates the smaller iteration budget for subsequent frames (Section 3.2).

Implicit temporal coupling. For a locally quadratic objective, finite-step optimization retains a dependence on its initialization $z ^ { ( 0 ) }$ . Hence, SUP with a finite budget and early stopping implicitly biases $z _ { t } ^ { * }$ toward $z _ { t - 1 } ^ { * } ,$ favoring temporally coherent reconstructions without an explicit temporal penalty or motion estimation, while the perturbation $\sigma n _ { t }$ keeps the optimization from being locked in the previous frame-specific content, such as wall and valve motion, which can be fitted.

## 2.5 Early Stopping and Algorithm

Iteratively fitting a highly expressive generator eventually reproduces measurement noise, as observed for DIP [26], so we stop each frame when the objective plateaus. Let $\dot { \mathcal { L } } ^ { ( k ) }$ be the objective of (9) at iteration k and ${ \mathcal { L } } _ { \mathrm { b e s t } }$ the lowest value observed so far. After a warm-up of $K _ { \mathrm { m i n } }$ iterations, a counter is incremented whenever $\mathcal { L } ^ { ( k ) } \geq ( 1 - \epsilon _ { s } ) \mathcal { L } _ { \mathrm { b e s t } }$ and reset otherwise. Optimization stops when the counter reaches the patience $P$ or after $K _ { \mathrm { m a x } }$ iterations. Because SUP provides a warm start, subsequent frames use smaller $K _ { \mathrm { m i n } }$ and $K _ { \mathrm { m a x } } ^ { \ast }$ than the first frame. The complete procedure is summarized in Algorithm 1.

## 3 Experiments

We evaluate UltraDiff in terms of restoration quality and robustness on simulated LR–HR echocardiography pairs, component contributions through ablations, and downstream utility through zero-shot EF prediction on real clinical videos.

## 3.1 Datasets

For restoration evaluation, we use the EchoNet-LVH dataset [9], which contains high-resolution echocardiography videos with a resolution of $1 0 2 4 \times 7 6 8$ pixels and 90–200 frames per clip. Each video is cropped and resized to 256×256, and the resulting frames are treated as reference HR videos. Since physically paired LR–HR ultrasound videos are rarely available, we follow the common full-reference evaluation protocol in ultrasound SR studies [22, 24, 25, 29]: LR observations are generated using $2 \times$ bicubic downsampling, followed by multiplicative Rayleigh speckle and additive Gaussian noise with a shared standard deviation varied from 0.02 to 0.10 to control the overall degradation strength. The same degraded LR inputs and HR references are used for all competing methods, ensuring a fair comparison under identical input conditions. Restoration quality is measured with the full-reference metrics PSNR, SSIM [37], and LPIPS [38], and the no-reference metrics NIQE [39], PIQE [40], and TOPIQ-NR [41].

![](images/3821a6f14b5c0ce17ad11fd99ee6b16de50e27f2310365ca4be6a1a7ecf31d03.jpg)

![](images/253c420743802f2e615008a3cbc66d9b1f40489c64f19b425b07e46cf3e49e3f.jpg)  
Figure 3: Qualitative comparison for 2× ultrasound video super-resolution. The left panel shows the HR reference and the simulated LR input. The top row compares SR outputs from different methods, with PSNR/SSIM reported below each result, and the bottom row shows the corresponding error maps with zoomed regions for structural inspection.  
Figure 4: Robustness analysis under increasing mixed-noise levels from 0.02 to 0.10. Box-and-violin plots report PSNR and SSIM across different methods. UltraDiff maintains higher medians and more concentrated distributions, indicating stronger restoration stability under heavier degradation.

For downstream evaluation, we use the EchoNet-Dynamic dataset [36], which contains real clinical echocardiography videos of 112×112 pixels and 28–1002 frames per clip, together with reference EF labels. We select 100 videos using EF-stratified sampling so that the subset follows the EF distribution of the full dataset. This evaluation does not rely on synthetic degradations or HR references. Instead, it tests whether restoration improves functional measurement on real clinical videos, where the true degradation is unknown.

## 3.2 Implementation Details

Our main UltraDiff configuration uses a CelebA-HQ pre-trained diffusion model as the frozen generative prior. This default is deliberate: the prior has never seen US data, so improvements can be attributed to the proposed formulation rather than to domain-specific training or any overlap between the training data, allowing us to test whether the proposed test-time optimization can adapt a generic image SR prior to ultrasound restoration. Unless otherwise noted, UltraDiff refers to this CelebA-HQ-prior configuration, which is used for ablation studies and downstream EF prediction.

Table 1: Quantitative restoration performance on EchoNet-LVH. UltraDiff variants use different pre-trained diffusion priors. The best, second-best, and third-best results are highlighted in red, orange, and blue, respectively.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>PIQE↓</td><td>TOPIQ-NR↑</td></tr><tr><td>Bicubic</td><td>21.76</td><td>0.5323</td><td>0.1721</td><td>11.0904</td><td>43.7747</td><td>0.4173</td></tr><tr><td>DiffVSR [32]</td><td>23.80</td><td>0.7395</td><td>0.1011</td><td>9.4150</td><td>47.1095</td><td>0.4651</td></tr><tr><td>DUP [29]</td><td>23.67</td><td>0.8096</td><td>0.1665</td><td>10.0552</td><td>48.7218</td><td>0.3836</td></tr><tr><td>EDSR [18]</td><td>21.87</td><td>0.7573</td><td>0.2193</td><td>17.3222</td><td>57.1770</td><td>0.4848</td></tr><tr><td>ZSSR-Cycle [25]</td><td>22.91</td><td>0.7894</td><td>0.2027</td><td>17.5018</td><td>50.1141</td><td>0.4431</td></tr><tr><td>UltraDiff-DFUI</td><td>25.27</td><td>0.8229</td><td>0.1843</td><td>7.8461</td><td>40.4669</td><td>0.4624</td></tr><tr><td>UltraDiff-ZEA</td><td>25.62</td><td>0.8484</td><td>0.0909</td><td>6.9604</td><td>41.2375</td><td>0.5000</td></tr><tr><td>UltraDiff-CelebA-HQ</td><td>25.04</td><td>0.8209</td><td>0.1101</td><td>9.1370</td><td>36.2833</td><td>0.4920</td></tr></table>

To examine the sensitivity to the choice of frozen diffusion prior, we also instantiate the same UltraDiff optimization framework with two additional priors, denoted UltraDiff-DFUI and UltraDiff-ZEA, in the restoration comparison.

During test-time optimization, a DDIM sampler [30] maps each latent seed to an HR estimate, while the diffusion model weights remain fixed and only the latent seed is optimized. The degradation operator A is implemented with bicubic downsampling, and the regularization weight is set to λ = 0.001 in Eq. 9. For the seed update strategy, the perturbation noise level is fixed at 0.1.

For the first frame of each video, we run up to 3000 optimization steps and enable early stopping after 2000 steps with patience 100. For subsequent frames, we reduce the maximum number of optimization steps to 2000 and enable early stopping after 800 steps with the same patience, since these frames are initialized from the propagated seed. For downstream evaluation, we use EchoCLIP [10] in a zero-shot setting to predict EF from SR videos. All input frames are resized to 224 × 224 before inference. All experiments are implemented in PyTorch and conducted on a single NVIDIA GeForce RTX 3090 GPU with 24 GB memory. Baselines are evaluated on the same LR inputs using their public settings or official implementations.

## 3.3 Comparison with Baselines

We compare UltraDiff with five representative baselines from four groups: Bicubic interpolation, learning-based single-image SR (EDSR [18]), video-based SR/restoration (DiffVSR [32] and DUP [29]), and ultrasound-specific SR (ZSSR-Cycle [25]). This set covers interpolation, generic image SR, video restoration, and ultrasound-oriented SR. All methods use the same LR inputs and HR references, so differences reflect restoration behavior rather than degradation settings.

Fig. 3 shows a qualitative comparison on a 2× SR example corrupted by mixed noise with std of 0.04. UltraDiff reconstructs clearer ventricular structures and produces lower errors in the zoomed regions than the baselines. For ultrasound restoration, improved sharpness should not come at the cost of distorted anatomical boundaries or unrealistic speckle texture.

The quantitative results in Table 1 further support the visual comparison. The main UltraDiff-CelebA-HQ setting achieves 25.04 PSNR and 0.8209 SSIM, outperforming all non-UltraDiff baselines on both metrics. This result sug gests that the proposed test-time diffusion-prior optimization can adapt a frozen image SR prior to echocardiography restoration, even when the prior is not trained specifically on ultrasound videos. The additional UltraDiff-DFUI and UltraDiff-ZEA variants use the same optimization framework with DFUI [42] and ZEA [43] frozen priors, respectively, with ZEA achieving the strongest restoration scores. This shows that UltraDiff is not tied to a single checkpoint, while CelebA-HQ remains the default setting for ablation and downstream analyses.

## 3.4 Robustness and Runtime Analysis

To evaluate robustness under different degradation strengths, Fig. 4 reports PSNR and SSIM as the mixed-noise level increases from 0.02 to 0.10. UltraDiff maintains higher reconstruction quality and a more concentrated distribution, suggesting that the speckle-aware fidelity term and diffusion prior stabilize restoration under stronger degradation.

Table 2: Ablation study of UltraDiff components on restoration quality. Removing each component degrades performance.
<table><tr><td>Variations</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>PIQE↓</td></tr><tr><td>MSE fidelity</td><td>24.42</td><td>0.8087</td><td>0.1233</td><td>9.7698</td><td>38.0033</td></tr><tr><td>w/o TV</td><td>24.47</td><td>0.8076</td><td>0.1210</td><td>9.4159</td><td>38.6782</td></tr><tr><td>w/o SUP</td><td>24.13</td><td>0.7636</td><td>0.2934</td><td>11.2284</td><td>76.2607</td></tr><tr><td>UltraDiff</td><td>25.04</td><td>0.8209</td><td>0.1101</td><td>9.1370</td><td>36.2833</td></tr></table>

Table 3: Downstream zero-shot EF prediction on the 100-video EchoNet-Dynamic subset after SR. MAE (EF %), Acc@±5, and Recall $@ 5 _ { \pm 5 }$ are reported. UltraDiff denotes the default CelebA-HQ setting.
<table><tr><td>Method</td><td>MAE↓</td><td>Acc@±5↑</td><td> $\mathbf { R e c a l l } @ 5 _ { \pm 5 }$  个</td></tr><tr><td>Bicubic</td><td>11.00</td><td>28.00</td><td>67.00</td></tr><tr><td>EDSR</td><td>10.60</td><td>26.00</td><td>58.00</td></tr><tr><td>ZSSR-Cycle</td><td>8.07</td><td>37.00</td><td>72.00</td></tr><tr><td>DiffVSR</td><td>8.35</td><td>36.00</td><td>60.00</td></tr><tr><td>DUP</td><td>7.31</td><td>44.00</td><td>72.00</td></tr><tr><td>UltraDiff</td><td>6.37</td><td>53.00</td><td>74.00</td></tr></table>

UltraDiff is a test-time optimization method intended for offline restoration. Under the setup in Section 3.2, it restores a frame in about 83 s on average, comparable to DUP (77 s) and faster than DiffVSR (93 s) and ZSSR-Cycle (152 s), but slower than feed-forward baselines such as Bicubic and EDSR. The cost is dominated by per-frame latent optimization, while SUP warm-start and early stopping help reduce the cost for subsequent frames. This trade-off is suitable for retrospective EF analysis, where offline processing is acceptable.

## 3.5 Ablation Studies

We conduct ablation studies to isolate the contribution of each component in UltraDiff. Specifically, we replace the proposed fidelity term with an MSE-based fidelity, remove TV regularization, and disable the seed update strategy. As shown in Table 2, each modification degrades performance, confirming that the fidelity term, regularization, and temporal seed propagation all contribute to restoration quality.

Replacing $\ell _ { s }$ with the MSE reduces PSNR from 25.04 to 24.42 dB and increases LPIPS from 0.1101 to 0.1233, consistent with Propositions 1 and 2: the homoscedastic MSE over-weights bright, high-variance regions, whereas $\ell _ { s }$ measures residuals relative to the local signal level. Removing TV yields a comparable PSNR drop (24.47 dB) and a higher PIQE, reflecting its role in promoting piecewise-constant structures: it suppresses residual fluctuations within homogeneous regions, such as the blood pool and myocardium, while preserving tissue boundaries. Disabling SUP causes the largest degradation, especially in SSIM, LPIPS, and PIQE: without the warm start, each frame starts from an unrelated seed whose initial estimate is far from the target, so the optimization is less likely to reach a good solution within the budget, in line with the bounded-discrepancy argument in Section 2.4.

## 3.6 Downstream Evaluations

Restoration metrics alone do not fully indicate whether an enhanced ultrasound video is useful for clinical analysis. We therefore evaluate all SR methods on the 100-video EchoNet-Dynamic subset for zero-shot ejection fraction (EF) prediction using EchoCLIP [10]. As reported in Table 3, we use mean absolute error (MAE) in EF percentage points, Acc@±5, and Recal $1 @ 5 _ { \pm 5 } .$ . Acc@±5 measures the percentage of predictions within ±5 EF points, while Recal $1 @ 5 _ { \pm 5 }$ measures whether one of the top-5 EF candidates falls within the same tolerance.

UltraDiff achieves the best performance across all three metrics, with an MAE of 6.37, Acc@±5 of 53.00%, and Recall $@ 5 _ { \pm 5 }$ of 74.00%. Compared with Bicubic interpolation, UltraDiff reduces MAE from 11.00 to 6.37 and improves Acc@±5 from 28.00% to 53.00%. It also reduces MAE by 12.9% over DUP, the strongest non-UltraDiff baseline. Fig. 5 further shows that UltraDiff achieves the highest or tied-highest Acc@k across error tolerances, indicating that the improvement is not limited to a single threshold. The Bland–Altman analysis in Fig. 6 shows the smallest mean bias (−0.59 EF points) and the narrowest 95% limits of agreement, indicating closer agreement between predicted and reference EF values in this evaluation.

![](images/334e5252f5f800420f1d1c32566f4d32170065ae36927e4c3e3c481025d2463c.jpg)  
Figure 5: Accuracy-error threshold curves for downstream EF prediction on the 100-video EchoNet-Dynamic subset. Acc@k measures the percentage of cases whose absolute EF error is within k EF points. Higher curves indicate more reliable EF estimation across different error tolerances.

Finally, Fig. 7 provides a representative end-diastolic/end-systolic (ED/ES) case. The original LR frames are overlaid with ground-truth left ventricular (LV) contours, and the corresponding LV regions are enlarged across methods. Since EF estimation depends on the structural change between ED and ES frames, a useful restoration method should enhance the LV region without distorting this temporal anatomical difference. UltraDiff provides a clearer LV region while better preserving the ED/ES structural difference, consistent with its improved downstream EF prediction performance.

## 4 Discussion

Role of the priors. The DM enters (8) not as an explicit density but through the range constraint $I \in { \mathcal { R } } ( { \mathcal { M } } )$ and the inductive bias of the pre-trained denoiser. TV is an explicit regularizer that favors piecewise-constant structures: it suppresses residual noise within homogeneous regions, such as the blood pool and myocardium, while preserving sharp tissue boundaries, and it stabilizes the ill-posed SR problem. The two priors are complementary: TV regularizes homogeneous regions, whereas fine tissue texture is supplied by the DM.

Choice of prior. Because K is small and z is optimized without constraint, R(M) is in practice not restricted to the images on which the DM was trained (e.g., faces for CelebA-HQ). Although natural and medical images differ greatly at the whole-image scale, the statistics of local features, such as heavy-tailed gradient distributions, are similar across domains, and diffusion-based reconstruction outside the training distribution is driven primarily by such local features [44]. Accordingly, DMs trained on natural images, including CelebA-HQ, outperform medical-image-trained DMs in out-of-distribution MRI and CT reconstruction [44] and enable training-free enhancement across medical modalities, including ultrasound [45]. This reliance on local regularity also mitigates hallucination of domain-specific content: global structures of the training domain are not imposed, because the data term ties $\boldsymbol { \mathcal { A } } \boldsymbol { \mathcal { M } } ( z )$ to the observation at every pixel while the prior mainly contributes local regularity. The natural-image default thus shows that the gains stem from the formulation rather than domain-specific training.

Extensions of the measurement model. UltraDiff operates on displayed intensities, which makes it directly applicable to standard clinical archives such as EchoNet without access to raw data. The log-residual form of the fidelity (Proposition 1(i)) is naturally compatible with the logarithmic display of US images, and the scale-family formulation accommodates other speckle statistics, e.g., for partially developed speckle [12], simply by changing f. The EF results on real videos, whose degradation is unknown, indicate robustness beyond the operator. Promising extensions include envelope or radio-frequency data where available, spatially correlated speckle models, and joint estimation of A for fully blind restoration.

Future directions. The modular design of UltraDiff allows each component to be upgraded independently. On the prior side, video DMs with built-in temporal modeling and ultrasound foundation DMs can be plugged in without retraining. On the regularization side, higher-order terms [46] and total generalized variation [47], which favor piecewise-smooth rather than piecewise-constant solutions, can further refine smoothly varying tissue regions. The per-frame optimization can be accelerated through diffusion distillation, consistency models, fewer sampling steps, or amortized seed prediction toward near-real-time operation. Building on the EF results reported here, future validation will cover higher scale factors, other anatomies such as obstetric and musculoskeletal US, dedicated temporal-consistency metrics, and clinical reader studies.

![](images/2863450af5fa20bcf3f38a61035423f69c447b6de7f5be35b61044b3ffce4124.jpg)

Figure 6: Bland–Altman analysis of downstream EF prediction on the 100-video EchoNet-Dynamic subset. The solid line denotes the mean prediction bias, and the dashed lines denote the 95% limits of agreement (bias ± 1.96 SD). UltraDiff shows the smallest bias and the narrowest agreement range among the compared SR methods.  
![](images/6e444e7222dca441d5a195ffcb0c2dadffa0ff126437ae02d88a4a975b10f99a.jpg)  
Figure 7: Representative end-diastolic/end-systolic (ED/ES) visualization for downstream EF prediction. The original LR frames are shown with ground-truth left ventricular (LV) contours, and the dashed boxes indicate the LV regions enlarged for each 2× SR method. UltraDiff provides a clearer LV region while better preserving the ED/ES structural difference.

## 5 Conclusion

We presented UltraDiff, a self-supervised US video restoration framework that uses a frozen pre-trained diffusion model as a plug-in prior. By optimizing latent seeds under a range-constrained MAP-inspired formulation with a speckle-aware, variance-adaptive fidelity and a temporal seed update, UltraDiff performs joint super-resolution and despeckling without paired HR data or model retraining. Experiments on two public echocardiography datasets demonstrate improved restoration quality, robustness to mixed noise, and better zero-shot EF prediction than representative SR baselines, supporting diffusion-prior-based self-supervised restoration for retrospective US video analysis.

## Acknowledgments

This work was supported in part by the Start-up Grant of City University of Hong Kong (Dongguan).

## References

[1] Islam Aly, Asad Rizvi, Wallisa Roberts, Shehzad Khalid, Mohammad W Kassem, Sonja Salandy, Maira du Plessis, R Shane Tubbs, and Marios Loukas. Cardiac ultrasound: an anatomical and clinical review. Transla tional Research in Anatomy, 22:100083, 2021.

[2] Pierpaolo Pellicori, Elke Platz, Jeroen Dauw, Jozine M Ter Maaten, Pieter Martens, Emanuele Pivetta, John GF Cleland, John JV McMurray, Wilfried Mullens, Scott D Solomon, et al. Ultrasound imaging of congestion in heart failure: examinations beyond the heart. European Journal ofHeart Failure, 23(5):703–712, 2021.

[3] Qianye Yang, Kangning Cui, Yipeng Hu, Can Peng, Netzahualcoyotl Hernandez-Cruz, Rahul Ahuja, Elena D’Alberti, Chan Raymond, Olga Patey, Aris Papageorghiou, et al. A deep learning framework for fetal heart tracking in ultrasound videos: Toward enhanced congenital heart defects detection. In International Conference on Functional Imaging and Modeling ofthe Heart, pages 219–230. Springer, 2025.

[4] Kwok-Yin Leung. Applications of advanced ultrasound technology in obstetrics. Diagnostics, 11(7):1217, 2021.

[5] Zarko Alfirevic, Tamara Stampalija, and Nancy Medley. Fetal and umbilical doppler ultrasound in normal pregnancy. Cochrane Database ofSystematic Reviews, 2015.

[6] Pravin Patil and Bhaskar Dasgupta. Role of diagnostic ultrasound in the assessment of musculoskeletal diseases. Therapeutic Advances in Musculoskeletal Disease, 4(5):341–355, 2012.

[7] Jung Hyun Yoon, Min Jung Kim, Hye Sun Lee, Sung Hun Kim, Ji Hyun Youk, Sun Hye Jeong, and You Me Kim. Validation of the fifth edition bi-rads ultrasound lexicon with comparison of fourth and fifth edition diagnostic performance using video clips. Ultrasonography, 35(4):318, 2016.

[8] Guojia Zhao, Dezhuag Kong, Xiangli Xu, Shunbo Hu, Ziyao Li, and Jiawei Tian. Deep learning-based classification of breast lesions using dynamic ultrasound video. European Journal of Radiology, 165:110885, 2023.

[9] Grant Duffy, Paul Cheng, Neal Yuan, Bryan He, Alan Kwan, Matthew Shun-Shin, Kevin Alexander, Joseph Ebinger, Matthew Lungren, Florian Rader, Ingela Schnittger, Euan Ashley, James Zou, Jignesh Patel, Ronald Witteles, Susan Cheng, and David Ouyang. High-throughput precision phenotyping of left ventricular hypertrophy with cardiovascular deep learning. JAMA Cardiology, 7(4):386–395, 2022.

[10] Matthew Christensen, Milos Vukadinovic, Neal Yuan, and David Ouyang. Vision–language foundation mode for echocardiogram interpretation. Nature Medicine, pages 1–8, 2024.

[11] Yingyu Yang, Qianye Yang, Kangning Cui, Can Peng, Elena D’Alberti, Netzahualcoyotl Hernandez-Cruz, Olga Patey, Aris T Papageorghiou, and J Alison Noble. Latent motion profiling for annotation-free cardiac phase detection in adult and fetal echocardiography videos. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 316–325. Springer, 2025.

[12] Anparasy Sivaanpu, Kumaradevan Punithakumar, Rui Zheng, Kim-Cuong T Nguyen, Michelle Noga, Dean Ta, Edmond HM Lou, and Lawrence H Le. Speckle noise reduction techniques in ultrasound imaging: A comprehensive review of the last two decades (2005–2024). Computer Methods and Programs in Biomedicine, page 109150, 2025.

[13] Oleg V Michailovich and Allen Tannenbaum. Despeckling of medical ultrasound images. ieee transactions on ultrasonics, ferroelectrics, and frequency control, 53(1):64–78, 2006.

[14] Linwei Yue, Huanfeng Shen, Jie Li, Qiangqiang Yuan, Hongyan Zhang, and Liangpei Zhang. Image superresolution: The techniques, applications, and future. Signal Processing, 128:389–408, 2016.

[15] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 126–135, 2017.

[16] Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung Mu Lee. Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 0–0, 2019.

[17] Ce Liu and Deqing Sun. On bayesian adaptive video super resolution. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(2):346–360, 2013.

[18] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 136–144, 2017.

[19] Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu. Residual dense network for image superresolution. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2472– 2481, 2018.

[20] Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, and William T Freeman. Video enhancement with taskoriented flow. International Journal of Computer Vision, 127:1106–1125, 2019.

[21] Xingyu Zhou, Leheng Zhang, Xiaorui Zhao, Keze Wang, Leida Li, and Shuhang Gu. Video super-resolution transformer with masked inter&intra-frame attention. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25399–25408, 2024.

[22] Woosuk Choi, Mina Kim, Jae HakLee, Jungho Kim, and Jong BeomRa. Deep cnn-based ultrasound superresolution for high-speed high-resolution b-mode imaging. In 2018 IEEE International Ultrasonics Symposium (IUS), pages 1–4, 2018.

[23] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image superresolution using a generative adversarial network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

[24] Ruud JG Van Sloun, Oren Solomon, Matthew Bruce, Zin Z Khaing, Yonina C Eldar, and Massimo Mischi. Deep learning for super-resolution vascular ultrasound imaging. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1055–1059. IEEE, 2019.

[25] Heng Liu, Jianyong Liu, Shudong Hou, Tao Tao, and Jungong Han. Perception consistency ultrasound image super-resolution via self-supervised cyclegan. Neural Computing and Applications, pages 1–11, 2021.

[26] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[27] Chenyang Lei, Yazhou Xing, Hao Ouyang, and Qifeng Chen. Deep video prior for video consistency and propagation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(1):356–371, 2022.

[28] Pasquale Cascarano, Maria Colomba Comes, Arianna Mencattini, Maria Carla Parrini, Elena Loli Piccolomini, and Eugenio Martinelli. Recursive deep prior video: a super resolution algorithm for time-lapse microscopy of organ-on-chip experiments. Medical Image Analysis, 72:102124, 2021.

[29] Chu Chen, Kangning Cui, Pasquale Cascarano, Wei Tang, Elena Loli Piccolomini, and Raymond H Chan. Blind restoration of high-resolution ultrasound video. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 77–87. Springer, 2025.

[30] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[31] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[32] Xiaohui Li, Yihao Liu, Shuo Cao, Ziyan Chen, Shaobin Zhuang, Xiangyu Chen, Yinan He, Yi Wang, and Yu Qiao. Diffvsr: Revealing an effective recipe for taming robust video super-resolution against complex degradations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15319–15328, 2025.

[33] Tianyu Liu, Shuai Han, Linru Xie, Wenyu Xing, Chengcheng Liu, Boyi Li, and Dean Ta. Super-resolution reconstruction of ultrasound image using a modified diffusion model. Physics in Medicine & Biology, 69(12):125026, 2024.

[34] Hengkang Wang, Xu Zhang, Taihui Li, Yuxiang Wan, Tiancong Chen, and Ju Sun. DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models, May 2024. arXiv:2405.16749 [cs].

[35] Leonid I Rudin, Stanley Osher, and Emad Fatemi. Nonlinear total variation based noise removal algorithms. Physica D: Nonlinear Phenomena, 60(1-4):259–268, 1992.

[36] David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curtis Langlotz, Paul Heidenreich, Robert Harrington, David Liang, Euan Ashley, and James Zou. Video-based ai for beat-to-beat assessment of cardiac function. Nature, 580, 04 2020.

[37] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[38] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[39] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.

[40] Narasimhan Venkatanath, D Praneeth, S Channappayya Sumohana, S Medasani Swarup, et al. Blind image quality evaluation using perception based features. In 2015 twenty first national conference on communications (NCC), pages 1–6. IEEE, 2015.

[41] Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Topiq: A top-down approach from semantics to distortions for image quality assessment. IEEE Transactions on Image Processing, 33:2404–2418, 2024.

[42] Marina Domínguez, Yordanka Velikova, Nassir Navab, and Mohammad Farid Azampour. Diffusion as sound propagation: Physics-inspired model for ultrasound image generation, 2024.

[43] Tristan S.W. Stevens, Wessel L. van Nierop, Ben Luijten, Vincent van de Schaft, Oisín Nolan, Beatrice Federici, Louis D. van Harten, Simon W. Penninga, Noortje I.P. Schueler, and Ruud J.G. van Sloun. zea: A toolbox for cognitive ultrasound imaging. Journal ofOpen Source Software, 11(121):9881, 2026.

[44] Lukas Glaszner, Martin Zach, and Thomas Pock. Bigger isn’t always better: Towards a general prior for medical image reconstruction. International Journal ofComputer Vision, 134(6):307, 2026.

[45] Ben Fei, Yixuan Li, Weidong Yang, Hengjun Gao, Jingyi Xu, Lipeng Ma, Yatian Yang, and Pinghong Zhou. A diffusion model for universal medical image enhancement. Communications Medicine, 5(1):294, 2025.

[46] Tony Chan, Antonio Marquina, and Pep Mulet. High-order total variation-based image restoration. SIAM Journal on Scientific Computing, 22(2):503–516, 2000.

[47] Kristian Bredies, Karl Kunisch, and Thomas Pock. Total generalized variation. SIAM Journal on Imaging Sciences, 3(3):492–526, 2010.