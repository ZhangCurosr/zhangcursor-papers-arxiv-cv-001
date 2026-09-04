# DSAQuant: Denoising-Stage-Aligned Quantization-Aware Training for Video Generation

Shuaiting Li<sup>1,2</sup> Zelin Gao<sup>1,4</sup> Haibin Shen<sup>2</sup> Yujun Shen<sup>1</sup>

Haotong Qin<sup>3†</sup>

<sup>1</sup>Robbyant <sup>2</sup>ZJU <sup>3</sup>PolyU <sup>4</sup>HKUST

<sup>†</sup>Corresponding Authors

## Abstract

Video diffusion models (VDMs) have achieved impressive progress in text-to-video generation, but their high memory and computational costs hinder practical deploy ment. Quantization-aware training (QAT) is an effective solution for compressing and accelerating advanced generative models without runtime overhead at inference. However, existing QAT methods suffer from a distinctive challenge in VDMs: while they often preserve prompt semantics, global layout, and coarse motion, the quantized model severely degrades visual details, texture fidelity, and sharpness. In this paper, we trace this degradation to the timestep-agnostic design of conventional quantization pipelines, which overlooks the stage-wise functionality of video denoising. In VDMs, early denoising steps mainly establish global structure and motion, whereas middle and late steps refine local appearance and high-frequency details. Based on this insight, we propose DSAQuant, a Denoising-Stage-Aligned Quantization-aware training framework for VDMs. During training, Denoising-Stage Oriented Supervision preserves teacher distillation in early steps for stable structure planning, while shifting later steps toward target-driven optimization to enhance detail reconstruction. During inference, Denoising-Stage Gated Guidance disables CFG in the final denoising steps to prevent it from amplifying quantizationinduced errors into high-frequency artifacts. Extensive experiments on the Wan and CogVideoX families under W4A4 and W3A3 settings show that DSAQuant consistently outperforms the SOTA QAT baseline, improving the VBench average score by up to 6.60 under aggressive W3A3 quantization while preserving strong text-video alignment. These results demonstrate that effective VDM quantization requires not only reducing quantization error, but also aligning quantization training and inference with the stage-wise nature of video diffusion.

Project page: https://robbyant-research.github.io/DSAQuant/ Code: https://github.com/robbyant-research/DSAQuant

## 1 Introduction

Video diffusion models (VDMs) [30, 27, 13] produce state-of-the-art text-to-video at compute and memory costs that block real deployment, making quantization a critical path to practical use. Existing post-training quantization (PTQ) methods [33, 14, 1] rely on online transformations whose per-layer runtime overhead on high-dimensional video tokens largely offsets the low-bit speedup, leaving advanced VDMs with limited practical benefit from low-bit PTQ.

Quantization-aware training (QAT) sidesteps this overhead by directly optimizing the model under low-bit constraints, with no auxiliary activation transformation at inference time. However, the only existing QAT framework tailored for VDMs [9] employs a timestep-agnostic MSE distillation pipeline, which applies the same teacher-imitation objective and the same classifier-free guidance (CFG) schedule across every denoising step. Yet video denoising itself is not timestep-agnostic: early steps establish global layout and motion, whereas middle and late steps refine textures and frame-level sharpness. Figure 1 makes the consequence concrete: a conventional quantized VDM still follows the prompt and reproduces the global layout (Fig. 1a), yet its imaging-quality and sharpness scores collapse while prompt-alignment scores stay close to FP (Fig. 1a), and a stage-wise substitution experiment localizes the challenge to the middle and late timesteps (Fig. 1b). A timestep-agnostic quantization strategy is thus structurally mismatched to VDMs.

We trace this mismatch to two timestep-agnostic design choices. On the training side, a low-bit student forced to imitate the FP teacher in late steps may spend scarce discrete capacity matching the teacher’s continuous-space fitting residual, without improving its own target prediction for detail reconstruction. On the inference side, the FP-level conditional and unconditional predictions converge in late steps, so the true guidance signal vanishes, while the two branches are quantized through independent activation grids whose errors are only weakly aligned and tend to accumulate rather than cancel; CFG then multiplicatively amplifies this uncorrelated quantization noise into the high-frequency artifacts we observe.

Building on this two-sided diagnosis, we propose DSAQuant, a denoising-stage-aligned QAT framework with two complementary components, one for each diagnosed failure. Denoising-Stage Oriented Supervision addresses the training-side capacity drain: we replace uniform MSE distillation with an exponential schedule that smoothly transfers supervision from a pure KD objective at t = 0 to a pure target-loss objective at t = 1. Early steps, therefore, inherit KD’s variance reduction against the noisy STE gradient and stably anchor global structure, while late steps decouple the quantized student from the FP teacher’s continuous-space fitting residual, freeing discrete capacity for genuine detail reconstruction. Denoising-Stage Gated Guidance addresses the inference-side noise amplification: we disable CFG in the final few denoising steps, removing the multiplicative amplifier that converts weakly aligned cond/uncond quantization errors into high-frequency artifacts, while preserving the strong text-video alignment that CFG provides at earlier steps. The two changes are orthogonal and complementary: the training-side change gives the quantized model the capacity to produce correct details, while the inference-side change ensures those details survive guidance. We validate DSAQuant across diverse VDMs (CogVideoX-2B/1.5-5B, Wan2.1-1.3B/14B, Wan2.2-5B) under both W4A4 and W3A3, achieving consistent and substantial improvements over the SOTA QAT baseline QVGen.

Our contributions are summarized as follows:

• We identify and empirically localize a stage-dependent failure mode in quantized VDMs: timestep-agnostic quantization preserves prompt semantics but severely degrades the detailrefinement capability of middle and late denoising steps.

• We propose DSAQuant, a denoising-stage-aligned QAT framework with two complementary components: Denoising-Stage Oriented Supervision with an exponential schedule that decouples the quantized student from the FP teacher’s continuous-space residual and frees discrete capacity for late-step detail; and Denoising-Stage Gated Guidance, supported by a noise-amplification analysis showing that CFG multiplicatively amplifies the weakly aligned cond/uncond quantization errors in late timesteps.

• Extensive experiments on diverse VDMs (CogVideoX-2B/1.5-5B, Wan2.1-1.3B/14B, Wan2.2-5B) under W4A4 and W3A3 settings demonstrate consistent and substantial improvements over SOTA QAT baselines, validating the effectiveness of our framework.

## 2 Related Work

Video diffusion models. Video Diffusion models (VDMs) based on diffusion transformer [20] architectures have achieved remarkable success in high-quality video generation [30, 27, 13, 35]. However, advanced video DiTs often contain billions of parameters and require multi-step denoising over long spatio-temporal sequences, leading to substantial latency and memory costs. To improve efficiency, prior works have explored step distillation [31, 18], sparse attention [29], step caching [19, 38, 36] and quantization [33, 4, 9].

![](images/05df3f7dda0b34fa39917104538ff4fd8f38d1df48dce88f0dd7352987cdd5a7.jpg)  
Figure 1: Analysis of conventional QAT on VDMs (please zoom in for details). (a) Conventional methods generally preserve layout and motion, but suffer severe degradation in details and texture. (b) We apply quantization to different denoising stages of VDMs: quantizing early steps slightly perturbs layout but preserves perceptual quality, while quantizing late steps significantly damages visual quality.

Quantization for diffusion models. Quantization [12] reduces model size and inference cost by representing weights and activations with low-bit values. Most video diffusion quantization methods focus on post-training quantization (PTQ). Q-DiT [2] applies PTQ to VDMs with dynamic activation quantization, and ViDiT-Q [33] further improves PTQ with mixed precision. S2Q-VDiT [4] leverages salient data and sparse token distillation. However, recent PTQ methods can still suffer from quality loss under activation quantization, and rotation-based transformations may introduce non-negligible runtime overhead in DiTs. QVGen [9] first explored quantization-aware training (QAT) for VDMs, showing that previous QAT methods on image diffusion [5, 17] can be applied to VDMs, and further improves weight quantization with auxiliary modules. We focus on the most relevant works in the main text and provide a more comprehensive discussion in Appendix A.

## 3 Preliminary

## 3.1 Video Diffusion Sampling

Video diffusion models (VDMs) generate videos through an iterative denoising process. Let $x _ { 1 }$ be the clean video latent and $\epsilon \sim \mathcal { N } ( 0 , I )$ be Gaussian noise. We use $t \in [ 0 , 1 ]$ to denote the sampling progress from noise to data, and write the noising path as

$$
x _ { t } = \alpha _ { t } x _ { 1 } + \sigma _ { t } \epsilon ,\tag{1}
$$

where $\alpha _ { 0 } = 0 , \sigma _ { 0 } = 1 , \alpha _ { 1 } = 1$ , and $\sigma _ { 1 } = 0$ . This notation includes DDIM-style sampling by choosing $\alpha _ { t } = \sqrt { \bar { \alpha } _ { \tau ( t ) } }$ and $\sigma _ { t } = \sqrt { 1 - \bar { \alpha } _ { \tau ( t ) } }$ , and flow matching by choosing, $\mathrm { e . g . , } \alpha _ { t } = t$ and $\sigma _ { t } = 1 - t .$

Under this formulation, the original training target can be written generically as

$$
y _ { t } = a _ { t } x _ { 1 } + b _ { t } \epsilon ,\tag{2}
$$

At each step, the denoising network tries to predict a diffusion target $f ( x _ { t } , c , t )$ , where the target can be noise, velocity, or a data-related quantity.

## 3.2 Stage-Wise Quantization Analysis for Video Denoising

The denoising process of VDMs is stage-dependent: with signal-to-noise ratio $\mathrm { S N R } _ { t } = \alpha _ { t } ^ { 2 } / \sigma _ { t } ^ { 2 }$ , early steps have low SNR and rely on the text condition and generative prior to establish global semantics, scene layout, and coarse motion, while late steps operate on latents that already carry recognizable structure and focus on refining local appearance, high-frequency textures, and frame-level sharpness.

Despite this stage-wise functionality, conventional quantization-aware training (QAT) treats every denoising step uniformly [5, 17, 9]: it optimizes a quantized model $f ^ { q }$ to match the full-precision model $f ^ { \mathrm { f p } }$ across all steps indiscriminately. However, as shown in Figure 1(a), directly applying such uniform QAT to VDMs leads to a distinctive degradation pattern: the quantized model preserves prompt semantics, global layout, and coarse motion, but suffers from blurred textures, corrupted details, reduced sharpness, and unstable local motion.

Denosing-Stage-Aligned Quantization-Aware Training  
![](images/208e7e9a177dd0ebde40ea64f8db35e87a0dc5a921d2adcbf391e964906f72c1.jpg)  
Figure 2: Overview of DSAQuant. (a) During training, Denoising-Stage Oriented Supervision smoothly transfers supervision from KD in early steps to the target loss in late steps via an exponential schedule, freeing late-step capacity for detail reconstruction. (b) During inference, Denoising-Stage Gated Guidance disables CFG in the final few steps, removing the multiplicative amplifier of cond/uncond quantization mismatch.

To identify the source of this degradation, we perform a stage-wise quantization analysis (Figure 1(b)), in which different sampling stages are quantized separately while the remaining stages are kept fullprecision. Quantizing only the early steps slightly perturbs global layout but preserves perceptual quality, while quantizing the middle and late steps reproduces the full failure mode. Consistent with the stage-wise picture above, conventional QAT fails not because it destroys semantic generation, but because its timestep-agnostic objective is inadequate for the detail-sensitive late stages. These observations motivate a denoising-stage-aligned framework that adapts both training supervision and inference guidance to the role each sampling stage plays.

## 4 Method

The failure mode identified in Sec. 3.2 traces to two timestep-agnostic design choices: a uniform distillation objective during training, and a fixed CFG scale during inference. To address them, we present DSAQuant, a Denoising-Stage-Aligned QAT framework with two stage-aware components: Denoising-Stage Oriented Supervision (Sec. 4.1) and Denoising-Stage Gated Guidance (Sec. 4.2).

## 4.1 Training Stage: Denoising-Stage Oriented Supervision

The standard target-driven loss used in pretraining of diffusion models is:

$$
\mathcal { L } _ { \mathrm { T a r g e t } } ( t ) = \left\| f ^ { q } ( x _ { t } , c , t ) - y _ { t } \right\| _ { 2 } ^ { 2 } .\tag{3}
$$

Instead, traditional QAT paradigms [5, 9, 17] usually train the quantized model by imitating the full-precision model, using distillation loss:

$$
\mathcal { L } _ { \mathrm { K D } } ( t ) = \Vert f ^ { q } ( x _ { t } , c , t ) - f ( x _ { t } , c , t ) \Vert _ { 2 } ^ { 2 } .\tag{4}
$$

In early high-noise steps, the noisy latent $x _ { t }$ contains limited information about the clean target $x _ { 1 }$ . In this regime, teacher imitation is useful in two complementary ways. First, constraining the quantized model to follow the full-precision teacher preserves functional similarity in semantic control, spatial layout, and coarse motion, thereby anchoring the global denoising trajectory. Second, the distillation target provides a lower-variance supervision signal than the sample-wise native target: it replaces the uncertain target $y _ { t }$ with the teacher’s estimate of the conditional denoising direction, which stabilizes STE-based QAT updates. We analyze this variance-reduction effect in Appendix B.

However, uniform distillation becomes less suitable in late denoising steps. At this stage, the global structure has largely been formed, and the model mainly refines local appearance, high-frequency textures, object boundaries, and frame-level sharpness. Under low-bit quantization, forcing it to imitate the teacher at every step can therefore waste its limited capacity on matching the teacher residuals rather than minimizing the true diffusion or flow target error. Since late-stage errors directly manifest as blur, texture corruption, and loss of sharpness, the supervision should be oriented toward detail refinement.

We therefore propose Denoising-Stage Oriented Supervision, a smooth denoising-stage hybrid objective that preserves teacher distillation in early steps and gradually shifts later steps toward the original diffusion or flow target. Formally, we define the overall training objective as a time-dependent weighted sum of the distillation loss and the native target loss:

$$
\mathcal { L } _ { \mathrm { m i x } } ( t ) = \lambda _ { \mathrm { K D } } ( t ) \mathcal { L } _ { \mathrm { K D } } ( t ) + \lambda _ { \mathrm { T a r g e t } } ( t ) \mathcal { L } _ { \mathrm { T a r g e t } } ( t )\tag{5}
$$

We design a normalized exponential scheduler to smoothly govern the transition. The weighting functions are formulated as:

$$
\lambda _ { \mathrm { K D } } ( t ) = \frac { \exp ( \alpha ( 1 - t ) ) - 1 } { \exp ( \alpha ) - 1 } , \quad \lambda _ { \mathrm { T a r g e t } } ( t ) = 1 - \lambda _ { \mathrm { K D } } ( t )\tag{6}
$$

Here, $\alpha > 0$ controls the strength of the exponential decay. By construction, $\lambda _ { \mathrm { K D } } ( 0 ) = 1$ and $\lambda _ { \mathrm { T a r g e t } } ( 1 ) = 1$ : early steps are fully driven by KD, anchoring layout and stabilizing the STE gradient, while late steps are driven by the target loss, freeing discrete capacity for high-frequency detail.

## 4.2 Inference Stage: Denoising-Stage Gated Guidance

Classifier-free guidance (CFG) is commonly used during inference to improve text-video alignment. We write CFG in the equivalent form

$$
\hat { f } _ { \omega } ( x _ { t } , c , t ) = f ( x _ { t } , \emptyset , t ) + \omega \left( f ( x _ { t } , c , t ) - f ( x _ { t } , \emptyset , t ) \right) ,\tag{7}
$$

where $\omega$ is the guidance strength. Let $y _ { t } = y _ { t } ( x _ { 1 } , \epsilon , t )$ denote the sample-wise original diffusion or flow target. The target is shared by the conditional and unconditional branches. We denote the optimal conditional and unconditional predictors as

$$
m _ { c } ( t ) = \mathbb { E } [ y _ { t } \mid x _ { t } , c ] , \qquad m _ { \emptyset } ( t ) = \mathbb { E } [ y _ { t } \mid x _ { t } ] .\tag{8}
$$

For the quantized model, we decompose the two predictions as

$$
f _ { c } ^ { q } ( t ) = m _ { c } ( t ) + \eta _ { c } ^ { q } ( t ) , \qquad f _ { \varnothing } ^ { q } ( t ) = m _ { \varnothing } ( t ) + \eta _ { \varnothing } ^ { q } ( t ) ,\tag{9}
$$

where $\eta _ { c } ^ { q } ( t )$ and $\eta _ { \varnothing } ^ { q } ( t )$ denote quantization-induced deviations from the ideal predictors.

Substituting Eq. (9) into Eq. (7), the guided prediction error with respect to the true target becomes

$$
\hat { f } _ { \omega } ^ { a } ( t ) - y _ { t } = \underbrace { m _ { \vartheta } ( t ) - y _ { t } } _ { \mathrm { t a r g e t r e s i d u a l } } + \underbrace { \eta _ { \vartheta } ^ { a } ( t ) } _ { \mathrm { u n c o n d q u a n t e r o r } } + \omega \Big ( \underbrace { m _ { c } ( t ) - m _ { \vartheta } ( t ) } _ { \mathrm { i d e a l g u i d a n c e d i r e c t i o n } } + \underbrace { \eta _ { c } ^ { q } ( t ) - \eta _ { \vartheta } ^ { q } ( t ) } _ { \mathrm { b r a r s h \ : q u a n t i z a t i o n \ : m i s m a t e n } } \Big ) .\tag{10}
$$

This decomposition shows that CFG amplifies not only the ideal guidance direction, but also the mismatch between the conditional and unconditional quantization errors. In quantized VDMs, the two branches can induce different activation distributions, leading to different clipping and rounding with dynamic activation quantization. Therefore, the quantization errors from the two branches are only weakly correlated; their difference tends to accumulate rather than cancel, and CFG amplifies this mismatch (analysis in Appendix C).

Early denoising benefits from CFG for prompt alignment and global structure formation. As the denoising process progresses, $x _ { t }$ encapsulates increasingly rich video details, leading to a relative decrease in shared semantic information; consequently, the ideal guidance gradually diminishes. However, as $x _ { t }$ deviates further from the initial Gaussian distribution, the quantization error continuously amplifies. As a result, the branch mismatch paradoxically emerges as the dominant factor in the trailing timesteps, which could lead to high-frequency artifacts. As evidenced in Fig. 3, the mismatch error rises rapidly in those late timesteps.

Table 1: Performance comparison across different quantization methods on VBench [11]. “\*” indicates results taken from the QVGen [9] paper. “<sup>†</sup>” denotes QVGen’s original asymmetric results from the public checkpoint. “Full Prec.” denotes the full-precision (BF16) model. Best results are highlighted in bold.
<table><tr><td>Method</td><td>#Bits (W/A)</td><td>Sym</td><td>Imaging↑ Quality</td><td>Aesthetic Quality</td><td>Motion Smoothness</td><td>Dynamic Degree</td><td>Background Consistency</td><td>Subject Consistency 个</td><td>Scene Consistency ←</td><td>Overall Consistency</td><td>Average ↑</td></tr><tr><td colspan="10">Wan2.1 1.3B (CFG = 5.0, 480p, fps = 16)</td></tr><tr><td>Full Prec.</td><td>| 16/16 |</td><td>-</td><td>64.22</td><td>57.85</td><td>97.25</td><td>71.11</td><td>95.72</td><td>93.34</td><td>26.07</td><td>24.70</td><td>66.28</td></tr><tr><td>ViDiT-Q*</td><td>4/6</td><td>×</td><td>56.24</td><td>50.18</td><td>94.81</td><td>52.43</td><td>89.67</td><td>82.53</td><td>13.45</td><td>19.58</td><td>57.36</td></tr><tr><td>SVDQuant*</td><td>4/6</td><td>√</td><td>58.16</td><td>51.27</td><td>97.05</td><td>49.44</td><td>93.74</td><td>91.71</td><td>14.18</td><td>23.26</td><td>59.85</td></tr><tr><td>LSQ*</td><td>4/4</td><td>× × ×</td><td>59.11</td><td>49.09</td><td>98.35</td><td>71.11</td><td>92.66</td><td>91.67</td><td>10.38</td><td>18.83</td><td>61.40</td></tr><tr><td>Q-DM*</td><td>4/4</td><td></td><td>60.40</td><td>52.50</td><td>97.22</td><td>76.67</td><td>93.37</td><td>89.26</td><td>13.28</td><td>21.63</td><td>63.04</td></tr><tr><td>EfficientDM*</td><td>4/4</td><td></td><td>60.70</td><td>53.57</td><td>96.18</td><td>56.39</td><td>93.74</td><td>91.70</td><td>11.77</td><td>21.19</td><td>60.66</td></tr><tr><td>QVGen†</td><td>4/4</td><td>× √</td><td>59.36</td><td>54.23</td><td>96.92</td><td>73.89</td><td>93.54</td><td>90.57</td><td>11.66</td><td>23.14</td><td>62.90</td></tr><tr><td>QVGen Ours</td><td>4/4 4/4</td><td>√</td><td>59.88 63.52</td><td>54.92 58.01</td><td>97.18 96.66</td><td>69.72 80.00</td><td>94.58 95.52</td><td>90.95 92.96</td><td>18.05 26.54</td><td>23.17 24.49</td><td>63.56 67.21</td></tr><tr><td colspan="10">CogVideoX-2B (CFG = 6.0, 480p, fps = 8)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>97.42</td><td>65.27</td><td>94.62</td><td>92.54</td><td>37.07</td><td></td><td></td></tr><tr><td>Full Prec.</td><td>16/16</td><td>一</td><td>58.21</td><td>54.37</td><td></td><td></td><td></td><td></td><td></td><td>25.10</td><td>65.57</td></tr><tr><td>ViDiT-Q* S2Q-VDiT*</td><td>4/6</td><td>×&gt;</td><td>54.72 53.71</td><td>43.01 52.31</td><td>92.18 98.09</td><td>43.22 36.11</td><td>90.76 96.15</td><td>81.02 93.99</td><td>26.25 34.23</td><td>20.41 24.90</td><td>56.45 61.18</td></tr><tr><td></td><td>4/6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LSQ*</td><td>4/4</td><td></td><td>58.73</td><td>54.20</td><td>97.57</td><td>45.00</td><td>92.97</td><td>92.41</td><td>24.06</td><td>23.17</td><td>61.01 61.48</td></tr><tr><td>Q-DM*</td><td>4/4 4/4</td><td>× × ×</td><td>54.96</td><td>52.71 51.97</td><td>98.00 98.03</td><td>48.61 46.67</td><td>93.82 94.10</td><td>91.86 91.70</td><td>28.02 27.76</td><td>23.87 24.28</td><td>61.31</td></tr><tr><td>EfficientDM* QVGen†</td><td>4/4</td><td></td><td>55.96 56.41</td><td>53.21</td><td>97.88</td><td>60.83</td><td>94.06</td><td>91.66</td><td>27.82</td><td>24.54</td><td>63.30</td></tr><tr><td>QVGen</td><td>4/4</td><td>× √</td><td>56.14</td><td>52.04</td><td>97.50</td><td>57.77</td><td>93.92</td><td>91.23</td><td>29.33</td><td>24.47</td><td>62.80</td></tr><tr><td>Ours</td><td>4/4</td><td>√</td><td>59.05</td><td>55.01</td><td>97.49</td><td>57.22</td><td>94.32</td><td>92.47</td><td>34.26</td><td>25.54</td><td>64.42</td></tr><tr><td colspan="10">Wan2.2 5B (CFG = 5.0, 720p, fps = 16)</td></tr><tr><td>Full Prec.</td><td></td><td></td><td>64.31</td><td>59.03</td><td>98.23</td><td>66.11</td><td>96.56</td><td>94.82</td><td>35.18</td><td>25.54</td><td>67.47</td></tr><tr><td>QVGen</td><td>|16/16</td><td></td><td></td><td>57.95</td><td></td><td></td><td></td><td>93.21</td><td>25.52</td><td></td><td></td></tr><tr><td>Ours</td><td>4/4 4/4</td><td>√ √</td><td>62.85 63.46</td><td>59.86</td><td>97.92 98.66</td><td>62.50 64.17</td><td>95.63 96.10</td><td>95.02</td><td>29.08</td><td>24.79 25.33</td><td>65.04 66.46</td></tr><tr><td colspan="10">CogVideoX1.5-5B (CFG = 6.0, 720p, fps = 16)</td></tr><tr><td>Full Prec.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>38.43</td><td></td><td></td></tr><tr><td>QVGen</td><td>|16/16</td><td></td><td>65.36</td><td>59.42</td><td>98.39</td><td>60.83</td><td>96.57</td><td>95.00</td><td></td><td>26.29</td><td>67.54</td></tr><tr><td>Ours</td><td>4/4</td><td>√</td><td>65.70</td><td>56.58</td><td>98.81</td><td>47.50</td><td>96.58 97.04</td><td>94.05 94.94</td><td>30.42 38.17</td><td>25.23 26.04</td><td>64.35 66.87</td></tr><tr><td></td><td>4/4</td><td>√</td><td>66.63</td><td>59.17</td><td>98.82</td><td>54.17</td><td></td><td></td><td></td><td></td><td></td></tr></table>

We therefore propose a gated guidance scheduler:

$$
\hat { f } _ { \omega } ^ { q } ( t ) = \left\{ \begin{array} { l l } { f _ { \varnothing } ^ { q } ( t ) + \omega \big ( f _ { c } ^ { q } ( t ) - f _ { \varnothing } ^ { q } ( t ) \big ) , } & { t \le \tau , } \\ { f _ { c } ^ { q } ( t ) , } & { t > \tau . } \end{array} \right.\tag{11}
$$

which removes the amplification of the branch quantization mismatch in the final few steps. (Computationally, the unconditional branch can also be skipped during these late steps, since it is no longer used.)

## 5 Experiments

![](images/75cd93597d9db6ebd4b47ffdbb7b9117796de4fb41f0d9a1e02ac119a0c8d662.jpg)  
Figure 3: Cond/uncond prediction MSE across denoising timesteps for FP vs. quantized models. While the FP model’s MSE stays near zero throughout, the quantized model’s MSE rises sharply in late timesteps (boxed region), showing that branch quantization mismatch dominates the (vanishing) ideal guidance signal.

## 5.1 Implementation details

Models and Baselines. Following QVGen [9], we conduct experiments on advanced open-source DMs, including CogVideoX-2B, CogVideoX1.5-5B [30], Wan2.1 1.3B and Wan2.1 14B. We further include Wan2.2 5B [27], a model with highly compressed VAE for fast generation. We adopt previous powerful PTQ and QAT methods as baselines: ViDiT-Q [33], SVDQuant [14], S2Q-VDIT [4], LSQ [3], Q-DM [17], EfficientDM [5], and QVGen [9]. We primarily compare our method against QVGen because it’s the first and only QAT framework tailored for VDMs, to our knowledge.

Quantization settings. Following QVGen [9], we use static per-channel weight quantization and dynamic per-token activation quantization. We use symmetric W4A4 quantization for kernel compatibility [14, 1] and additionally report asymmetric W3A3 results for aggressive low-bit settings. Because QVGen does not release its training data, we reproduce it with the same data, iterations, and quantizers as ours. Following [15, 23], we exclude timestep-conditioned linear layers from quantization because their finite-set inference outputs can be precomputed as LUTs (Appendix G).

Training. Following [7], we employ synthesized data based on text prompts from a filtered and LLM-extended version of VidProM [26] for training. Detailed settings can be found in Appendix E.

Table 2: Performance comparison on VBench [11] with augmented prompts under W4A4 and W3A3 settings. “<sup>†</sup>” denotes QVGen’s original asymmetric results from the public checkpoint. “Full Prec.” denotes the full-precision (BF16) model. Best results are highlighted in bold.
<table><tr><td>Method</td><td>#Bits (W/A)</td><td>Sym</td><td>|Imaging Quality</td><td>Aesthetic Quality</td><td>Motion Smoothness</td><td>Dynamic Background Degree Consistency</td><td>Subject Consistency</td><td>Scene ↑ Consistency</td><td>←</td><td>Overall</td><td>Average ↑</td></tr><tr><td colspan="10">CogVideoX 2B (CFG = 5.0, 480p, fps = 16)</td></tr><tr><td>Full Prec.</td><td>16/16 |</td><td></td><td>60.18</td><td>60.71</td><td>97.21</td><td>72.22</td><td>93.66</td><td>91.40</td><td>51.10</td><td>27.22</td><td>69.21</td></tr><tr><td>QVGen</td><td>4/4</td><td>√</td><td>60.53</td><td>59.87</td><td>97.15</td><td>64.80</td><td>93.43</td><td>90.41</td><td>50.07</td><td>27.13</td><td>67.94</td></tr><tr><td>Ours</td><td>4/4</td><td>√</td><td>61.86</td><td>62.37</td><td>97.36</td><td>61.28</td><td>94.12</td><td>90.73</td><td>53.34</td><td>27.34</td><td>68.55</td></tr><tr><td>QVGen</td><td>3/3</td><td>×</td><td>56.61</td><td>56.75</td><td>97.17</td><td>47.22</td><td>93.34</td><td>88.65</td><td>45.45</td><td>26.21</td><td>63.93</td></tr><tr><td>Ours</td><td>3/3</td><td>×</td><td>59.65</td><td>59.98</td><td>97.34</td><td>44.72</td><td>93.69</td><td>90.39</td><td>49.21</td><td>26.87</td><td>65.23</td></tr><tr><td colspan="10">Wan2.1 1.3B (CFG = 5.0, 480p, fps = 16)</td></tr><tr><td>Full Prec. |</td><td>16/16|</td><td></td><td>64.73</td><td>65.12</td><td>97.99</td><td>70.28</td><td>96.69</td><td>93.62</td><td>46.05</td><td>25.77</td><td>70.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>QVGen† QVGen</td><td>4/4 4/4</td><td>× √</td><td>62.93 58.42</td><td>63.25 59.50</td><td>98.30 97.21</td><td>62.77 63.06</td><td>95.75 94.56</td><td>93.13 89.43</td><td>40.45 32.50</td><td>25.04 24.14</td><td>67.71 64.85</td></tr><tr><td>Ours</td><td>4/4</td><td>√</td><td>63.99</td><td>64.36</td><td>97.95</td><td>73.33</td><td>96.30</td><td>93.17</td><td>44.10</td><td>25.62</td><td>69.85</td></tr><tr><td>QVGen</td><td>3/3</td><td>×</td><td>58.00</td><td>57.30</td><td>98.32</td><td></td><td>94.63</td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>3/3</td><td>×</td><td>62.50</td><td>62.27</td><td>97.96</td><td>35.55 68.05</td><td>95.54</td><td>90.24 92.09</td><td>32.00 37.73</td><td>22.57 25.30</td><td>61.08 67.68</td></tr><tr><td colspan="10">Wan2.2 5B (CFG = 5.0, 720p, fps = 16)</td></tr><tr><td>Full Prec. |</td><td>16/16</td><td></td><td>66.95</td><td>65.96</td><td>98.65</td><td></td><td>97.68</td><td>94.25</td><td>48.25</td><td>26.17</td><td>69.39</td></tr><tr><td>QVGen</td><td></td><td></td><td></td><td></td><td></td><td>57.22</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ôurs</td><td>4/4 4/4</td><td>了 √</td><td>64.84 66.06</td><td>63.68 66.36</td><td>98.49 98.86</td><td>52.22 55.27</td><td>96.75 97.70</td><td>93.78 94.75</td><td>44.68 46.21</td><td>25.43 25.86</td><td>67.48 68.88</td></tr><tr><td>QVGen</td><td>3/3</td><td></td><td>63.44</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>3/3</td><td>× ×</td><td>64.18</td><td>62.36 65.46</td><td>98.40 98.74</td><td>46.11 55.27</td><td>96.57 97.52</td><td>93.04 94.15</td><td>41.48 45.67</td><td>24.34 25.82</td><td>65.72 68.35</td></tr><tr><td colspan="10"></td></tr><tr><td>Full Prec. |</td><td>16/16 |</td><td></td><td>67.41</td><td>65.85</td><td>97.88</td><td>73.05</td><td>Wan2.1 14B (CFG = 5.0, 720p, fps = 16) 97.08</td><td>92.87</td><td>45.93</td><td>26.03</td><td>70.76</td></tr><tr><td>QVGen</td><td>4/4</td><td>√</td><td>66.51</td><td>65.35</td><td>97.84</td><td></td><td>96.14</td><td>92.59</td><td>41.96</td><td></td><td>68.34</td></tr><tr><td>Ours</td><td>4/4</td><td>√</td><td>67.60</td><td>65.88</td><td>98.38</td><td>62.16 68.89</td><td>97.33</td><td>93.75</td><td>43.69</td><td>25.15 26.04</td><td>70.20</td></tr><tr><td>QVGen</td><td>3/3</td><td>×</td><td>64.31</td><td>64.45</td><td>97.32</td><td>58.33</td><td>95.64</td><td>92.74</td><td>38.11</td><td>25.17</td><td>67.01</td></tr><tr><td>Ôurs</td><td>3/3</td><td>×</td><td>66.35</td><td>65.57</td><td>98.74</td><td>64.44</td><td>97.22</td><td>94.52</td><td>45.96</td><td>25.87</td><td>69.83</td></tr></table>

Table 3: Performance comparison on VBench-2.0 [34]. “<sup>∗</sup>” indicates results taken from the QVGen [9] paper. “<sup>†</sup>” denotes QVGen’s original asymmetric results from the public checkpoint. “Full Prec.” denotes the full-precision (BF16) model. Best results are highlighted in bold.
<table><tr><td>Method</td><td>#Bits (W/A) |Sym |Creativity↑ Commonsense↑ Control↑ Human Fidelity↑ Physics↑ Average↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Wan2.1 1.3B (CFG = 5.0, 480p, fps = 16)</td></tr><tr><td>Full Prec.</td><td>16/16</td><td></td><td>50.0</td><td>63.9</td><td>35.2</td><td>78.7</td><td>59.7</td><td>57.5</td></tr><tr><td>QVGen†</td><td>4/4</td><td>×</td><td>43.3</td><td>61.6</td><td>31.0</td><td>81.1</td><td>57.2</td><td>54.8</td></tr><tr><td>Ours</td><td>4/4</td><td>×</td><td>48.5</td><td>63.9</td><td>35.7</td><td>76.2</td><td>59.0</td><td>56.7</td></tr><tr><td>QVGen</td><td>4/4</td><td>√</td><td>46.5</td><td>57.5</td><td>28.0</td><td>70.1</td><td>57.2</td><td>51.9</td></tr><tr><td>Öurs</td><td>4/4</td><td>√</td><td>45.0</td><td>62.4</td><td>35.0</td><td>74.0</td><td>57.3</td><td>54.8</td></tr><tr><td>QVGen</td><td>3/3</td><td>×</td><td>37.3</td><td>51.9</td><td>23.6</td><td>72.3</td><td>52.6</td><td>47.5</td></tr><tr><td>Öurs</td><td>3/3</td><td>×</td><td>43.9</td><td>61.5</td><td>31.0</td><td>67.1</td><td>58.9</td><td>52.5</td></tr><tr><td colspan="9">Wan2.1 14B (CFG = 5.0, 720p, fps = 16)</td></tr><tr><td>Full Prec. |</td><td>16/16</td><td></td><td>55.2</td><td>64.0</td><td>37.3</td><td>81.6</td><td>62.8</td><td>60.2</td></tr><tr><td>QVGen*</td><td>4/4</td><td></td><td>48.0</td><td>55.5</td><td>40.0</td><td>80.5</td><td>62.2</td><td>57.2</td></tr><tr><td>QVGen</td><td>4/4</td><td>×&gt;</td><td>52.8</td><td>59.3</td><td>37.6</td><td>79.1</td><td>62.4</td><td>58.2</td></tr><tr><td>Òurs</td><td>4/4</td><td>√</td><td>54.6</td><td>63.4</td><td>37.8</td><td>81.3</td><td>62.4</td><td>59.9</td></tr><tr><td>QVGen</td><td>3/3</td><td>X</td><td>47.2</td><td>60.7</td><td>31.2</td><td>77.3</td><td>57.6</td><td>54.8</td></tr><tr><td>Ours</td><td>3/3</td><td>X</td><td>48.5</td><td>62.0</td><td>38.3</td><td>80.3</td><td>58.0</td><td>57.4</td></tr></table>

Evaluation. Our evaluation protocol incorporates multiple benchmarks. Following prior works [33, 4, 9], we evaluate 8 standard dimensions in VBench [11]. We additionally utilize VBench-2.0 [34] to gauge adherence to physical laws and reasoning. Furthermore, following [37], we measure Vision Reward and instruction following using high-motion prompts. Finally, the visual clarity of the videos is quantified via Laplacian Variance (LV).

## 5.2 Main results

Comparison with baselines. Tab. 1 presents the evaluation results on the standard VBench, alongside a comparison with existing PTQ and QAT frameworks. Under the 4-bit symmetric quantization setting, our method consistently outperforms QVGen and even surpasses its 4-bit asymmetric counterpart. Notably, on the Wan-1.3B model, our approach achieves a +8.49 improvement in scene consistency and a 3.64 gain in image quality. The consistent performance gains observed on the DDIM-based CogVideo further validate the generalizability of our method. Additional W3A3 VBench results are provided in Appendix H.

Table 4: Component ablation on Wan2.1-1.3B (W4A4, sym). We evaluate Denoising-Stage Oriented Supervision (mixed loss) and Denoising-Stage Gated Guidance (CFG-drop) independently. “Base” denotes the engineering-only quantized baseline with neither component. “LV” denotes Laplacian Variance. Best results are highlighted in bold; our full framework is shaded.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Mixed loss</td><td rowspan="2">CFG drop</td><td rowspan="2">#Bits (W/A)</td><td colspan="6">VBench1↑</td><td colspan="3">Visual Quality↑</td></tr><tr><td>Imaging Quality</td><td>Aesthetic Quality</td><td>Dynamic Degree</td><td>Scene Consistency</td><td>Overall Consistency</td><td>Avg</td><td>LV</td><td>Vision Reward</td><td>Instruc Following</td></tr><tr><td>Wan 1.3B (CFG = 5.0, 480p, fps = 16)</td><td colspan="10"></td><td rowspan="3"></td></tr><tr><td>FP</td><td></td><td></td><td>16/16</td><td>64.22</td><td>57.85</td><td>71.11</td><td>26.07</td><td>24.70</td><td>66.3</td><td>808</td></tr><tr><td>FP + CFG-drop</td><td>1 1</td><td>1√</td><td>16/16</td><td>63.57</td><td>57.99</td><td>70.56</td><td>27.50</td><td>24.74</td><td>66.4</td><td>850</td><td>7.51</td><td>37.5</td></tr><tr><td>Base</td><td>X</td><td>X</td><td>4/4</td><td>62.86</td><td>56.32</td><td>71.67</td><td>22.94</td><td>24.35</td><td>65.1</td><td>674</td><td>1.59</td><td>25.8</td></tr><tr><td>Base + CFG-drop</td><td>X</td><td>√</td><td>4/4</td><td>61.67</td><td>56.75</td><td>77.50</td><td>25.58</td><td>24.20</td><td>66.3</td><td>842</td><td>4.51</td><td>28.9</td></tr><tr><td>Base + Mixed loss</td><td>√</td><td>X</td><td>4/4</td><td>64.06</td><td>57.47</td><td>76.67</td><td>25.16</td><td>24.55</td><td>66.2</td><td>806</td><td>4.04</td><td>29.1</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>4/4</td><td>63.52</td><td>58.01</td><td>80.00</td><td>26.54</td><td>24.49</td><td>67.2</td><td>1026</td><td>6.47</td><td>34.5</td></tr></table>

Table 5: Ablation on the mixing schedule of Denoising-Stage Oriented Supervision (Wan2.1-1.3B, W4A4, sym). “linear” uses a linear schedule; “expα” uses the exponential schedule of Eq. 6 with parameter α. CFG-drop is applied by default. Our choice (exp5) is shaded; best results are in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2">#Bits (W/A)</td><td rowspan="2">Sym</td><td colspan="5">VBench1↑</td><td rowspan="2"></td><td colspan="3">Visual Quality↑</td></tr><tr><td>Imaging Quality</td><td>Aesthetic Quality</td><td>Dynamic Degree</td><td>Scene Consistency</td><td>Overall Consistency</td><td>Avg LV</td><td>Vision Reward</td><td>Instruc Following</td></tr><tr><td colspan="3">16/16</td><td></td><td></td><td>Wan 1.3B (CFG = 5.0, 480p, fps = 16)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="3">FP</td><td>一</td><td>64.22</td><td>57.85</td><td>71.11</td><td>26.07</td><td>24.70</td><td>66.3</td><td>808</td><td>7.51</td><td>37.5</td></tr><tr><td rowspan="4">Mix loss</td><td>linear</td><td>4/4</td><td>√</td><td>62.81</td><td>57.13</td><td>70.55</td><td>20.21</td><td>23.84</td><td>65.2</td><td>862</td><td>3.80</td><td>29.7</td></tr><tr><td>exp1</td><td>4/4</td><td>√</td><td>63.42</td><td>57.03</td><td>73.61</td><td>22.60</td><td>23.85</td><td>65.9</td><td>887</td><td>4.11</td><td>28.4</td></tr><tr><td>exp5</td><td>4/4</td><td>√</td><td>63.52</td><td>58.01</td><td>80.00</td><td>26.54</td><td>24.49</td><td>67.2</td><td>1026</td><td>6.47</td><td>34.5</td></tr><tr><td>exp10</td><td>4/4</td><td>√</td><td>63.50</td><td>57.76</td><td>74.70</td><td>18.79</td><td>23.43</td><td>65.6</td><td>785</td><td>4.19</td><td>24.8</td></tr></table>

Results on complex prompts and lower bitwidths. We further evaluate quantized models under augmented prompts, which contain more complex scenes and instructions. Results are summarized in Tab. 2. Across all tested Wan backbones, our W4A4 models consistently outperform prior quantization baselines and remain close to the full-precision models. For example, our method improves the average score over QVGen by 1.40–2.14 points at 4-bit, while reducing the gap to full precision to less than 0.6 points on all models. Moreover, our method remains robust under more aggressive 3-bit quantization, achieving 2.63–6.60 higher average scores than QVGen. Furthermore, we assess our models on VBench-2.0. As shown in Tab. 3, on Wan2.1-14B, our method maintains high performance even at 3-bit settings. These results demonstrate that our method preserves strong generation capability even for challenging prompts and lower bit-width settings. Representative qualitative comparisons are shown in Appendix K, with full video comparisons provided in the supplementary material.

## 5.3 Ablation study

To demonstrate the effect of each design, we employ W4A4 quantization on Wan 1.3B for ablation studies. Besides VBench, we also evaluate VisionReward and LV for a comprehensive comparison.

Ablation on two key designs. First, we evaluate the two proposed components in a 2-by-2 ablation, as summarized in Tab. 4. Denoising-Stage Oriented Supervision improves perceptual quality by aligning supervision with denoising stages, while Denoising-Stage Gated Guidance (CFG-drop) targets latestep guidance amplification. Combining the two components yields the best overall performance, suggesting that the training-side and inference-side designs are complementary. Representative qualitative comparisons are provided in the supplementary material.

Ablation on schedulers for Denoising-Stage-Oriented Supervision. Table 5 compares linear and exponential schedules for balancing distillation loss and target loss. The exponential schedule with α = 5 performs best, achieving the highest VBench average and visual quality scores. This suggests that a moderately sharp decay of the distillation loss better balances teacher guidance at early timesteps and target-loss optimization at later timesteps, while an overly aggressive decay with α = 10 harms performance.

Ablation on the threshold of Denoising-Stage-Gated Guidance. Fig. 4 illustrates the impact of the CFG-drop threshold on both visual clarity and VBench scores. We observe that applying CFG-drop even to only the final three timesteps yields a substantial boost in both metrics, effectively suppressing the amplification of quantization errors induced by CFG. However, when CFG-drop is extended beyond ten steps, the VBench score begins to decline, suggesting that premature removal of CFG compromises the semantic integrity of the generated video. Consequently, we set the threshold to nine steps, achieving a near-optimal balance between VBench performance and visual clarity.

Table 6: Training cost on four VDMs, reported in raw GPU-days. QVGen<sup>∗</sup> times are taken from the QVGen paper [9] (measured on H100); ours are measured on H20 under the settings in Appendix E. The per-row GPU is shown explicitly to avoid hardware-implicit comparison.
<table><tr><td>Method</td><td>GPU</td><td>CogVideoX-2B</td><td>Wan 1.3B</td><td>CogVideoX1.5-5B</td><td>Wan 14B</td></tr><tr><td> ${ \mathrm { Q V G e n } } ^ { * }$ </td><td>H100</td><td>9.44</td><td>11.11</td><td>~51</td><td>~182</td></tr><tr><td>Ours</td><td>H20</td><td>6.07</td><td>8.19</td><td>~65</td><td>~64</td></tr></table>

Table 7: Memory and inference latency on the Wan series, measured under 81 frames at 480p.
<table><tr><td>Model</td><td>Methods</td><td>Storage (GB)</td><td>Peak inference memory (GB)</td><td>A6000 latency (s)</td><td>4090D latency (s)</td></tr><tr><td>Wan 2.1-1.3B</td><td>FP16 Ours w4a4</td><td> $_ { 0 . 7 ( - 7 3 . 1 \% ) } ^ { 2 . 6 }$ </td><td> $_ { 7 . 0 ( - 1 9 . 5 \% ) } ^ { 8 . 7 }$ </td><td>313 264 (-15.7%)</td><td>246 193 (-21.5%)</td></tr><tr><td>Wan 2.1-14B</td><td>FP16 Ours w4a4</td><td>26.6 6.7 (-74.8%)</td><td>37.7 20.7 (-45.1%)</td><td>1845 1268 (-31.3%)</td><td>OOM 930</td></tr><tr><td>Wan 2.2-5B</td><td>FP16  $\mathrm { O u r s ~ w 4 a 4 }$ </td><td>9.5 2.4 (-74.7%)</td><td>16.9  $\bar { 1 0 . 9 } \ : ( - 3 5 . 5 \% )$ </td><td>129 80 (-38.0%)</td><td>100 58 (-42.0%)</td></tr></table>

## 5.4 Efficiency analysis

Training efficiency. Table 6 reports raw GPUdays together with the GPU type used for each method. In our training setup, H100 is approximately 2.5×–2.8× faster than H20<sup>1</sup>, so after normalizing both methods to H100-equivalent GPU-days, DSAQuant reduces training cost by roughly 2×–7×, with the largest gap on Wan 14B, indicating favorable scaling to large video diffusion models.

![](images/c0088605d284dd2c9f15844b35ca4424f2f1c54e17dffa29dc672bd4ffe64d42.jpg)  
Figure 4: Ablation on the threshold of Denoising-Stage Gated Guidance (CFG-drop).

Inference efficiency. Table 7 shows that our W4A4 model consistently improves inference

efficiency on the Wan series. Compared with FP16, it reduces storage by over 73%, peak memory by up to 45.1%, and A6000 latency by up to 38.0%. Notably, our quantized Wan 14B can run on 4090D, while the FP16 model leads to OOM. Moreover, our technique is orthogonal to attention acceleration methods, and combined results are provided in Appendix J.

## 6 Conclusion

We present DSAQuant, a denoising-stage aligned quantization-aware training framework for video diffusion models. Our key observation is that conventional QAT preserves coarse semantics but disrupts the detail-refinement capability of later denoising steps, resulting in degraded visual fidelity. DSAQuant addresses this issue through Denoising-stage-Oriented Supervision during training and Denoising-stage-Gated Guidance during inference, aligning both supervision and guidance with the stage-wise behavior of video denoising. Experiments demonstrate that DSAQuant improves visual quality, sharpness, and temporal stability while maintaining text-video alignment, highlighting the importance of stage-aware design for high-fidelity VDM quantization.

## Acknowledgments

We would like to thank Yichong Lu, Jiahao Wang, Yufeng Yuan, Nan Zhou, Jiahao Shao, and Yudong Jin for their insightful discussions.

## References

[1] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated llms, 2024.

[2] Lei Chen, Yuan Meng, Chen Tang, Xinzhu Ma, Jingyan Jiang, Xin Wang, Zhi Wang, and Wenwu Zhu. Q-dit: Accurate post-training quantization for diffusion transformers, 2024.

[3] Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S. Modha. Learned step size quantization. In International Conference on Learning Representations, 2020.

[4] Weilun Feng, Haotong Qin, Chuanguang Yang, Xiangqi Li, Han Yang, Yuqi Li, Zhulin An, Libo Huang, Michele Magno, and Yongjun Xu. S<sup>2</sup>Q-VDiT: Accurate quantized video diffusion transformer with salient data and sparse token distillation. arXiv preprint arXiv:2508.04016, 2025.

[5] Yefei He, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Efficientdm: Efficient quantization-aware fine-tuning of low-bit diffusion models. In The Twelfth International Conference on Learning Representations, 2024.

[6] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate post-training quantization for diffusion models, 2023.

[7] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

[8] Yushi Huang, Ruihao Gong, Jing Liu, Tianlong Chen, and Xianglong Liu. Tfmq-dm: Temporal feature maintenance quantization for diffusion models, 2024.

[9] Yushi Huang, Ruihao Gong, Jing Liu, Yifu Ding, Chengtao Lv, Haotong Qin, and Jun Zhang. QVGen: Pushing the limit of quantized video generative models. In The Fourteenth International Conference on Learning Representations, 2026.

[10] Yushi Huang, Ruihao Gong, Xianglong Liu, Jing Liu, Yuhang Li, Jiwen Lu, and Dacheng Tao. Temporal feature matters: A framework for diffusion model quantization, 2025.

[11] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

[12] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference, 2017.

[13] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models, 2025.

[14] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Junxian Guo, Xiuyu Li, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank component for 4-bit diffusion models. In The Thirteenth International Conference on Learning Representations, 2025.

[15] Shuaiting Li, Juncan Deng, Zeyu Wang, Kedong Xu, Rongtao Deng, Hong Gu, Haibin Shen, and Kejie Huang. Efficiency meets fidelity: A novel quantization framework for stable diffusion, 2025.

[16] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models, 2023.

[17] Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, and Baochang Zhang. Q-dm: An efficient lowbit quantized diffusion model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[18] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation, 2025.

[19] Zhengyao Lv, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K. Wong. Fastercache: Training-free video diffusion model acceleration with high quality. In The Thirteenth International Conference on Learning Representations, 2025.

[20] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[21] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models, 2023.

[22] Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, and Eunhyeok Park. Temporal dynamic quantization for diffusion models, 2023.

[23] Yang Sui, Yanyu Li, Anil Kag, Yerlan Idelbayev, Junli Cao, Ju Hu, Dhritiman Sagar, Bo Yuan, Sergey Tulyakov, and Jian Ren. Bitsfusion: 1.99 bits weight quantization of diffusion model. Advances in Neural Information Processing Systems, 37:76775–76818, 2024.

[24] Shilong Tian, Hong Chen, Chengtao Lv, Yu Liu, Jinyang Guo, Xianglong Liu, Shengxi Li, Hao Yang, and Tao Xie. Qvd: Post-training quantization for video diffusion models, 2024.

[25] Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards accurate post-training quantization for diffusion models, 2024.

[26] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-tovideo diffusion models. 2024.

[27] WanTeam, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025.

[28] Junyi Wu, Haoxuan Wang, Yuzhang Shang, Mubarak Shah, and Yan Yan. Ptq4dit: Post-training quantization for diffusion transformers, 2024.

[29] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, Jianfei Chen, Ion Stoica, Kurt Keutzer, and Song Han. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity, 2025.

[30] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer, 2025.

[31] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.

[32] Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Turbodiffusion: Accelerating video diffusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025.

[33] Tianchen Zhao, Tongcheng Fang, Haofeng Huang, Rui Wan, Widyadewi Soedarmadji, Enshu Liu, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, and Yu Wang. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation. In The Thirteenth International Conference on Learning Representations, 2025.

[34] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness, 2025.

[35] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024.

[36] Xin Zhou, Dingkang Liang, Kaijin Chen, Tianrui Feng, Xiwu Chen, Hongkai Lin, Yikang Ding, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. Less is enough: Training-free video diffusion acceleration via runtime-adaptive caching. arXiv preprint arXiv:2507.02860, 2025.

[37] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation, 2026.

[38] Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. In The Thirteenth International Conference on Learning Representations, 2025.

## A More Related Work

Quantization for image diffusion models. Early diffusion quantization studies mainly focus on image generation. PTQ methods such as Q-Diffusion [16], PTQD [6], and related works [21, 25] show that diffusion models require timestep-aware calibration because activation distributions vary substantially along the denoising trajectory. Temporal Dynamic Quantization [22], TFMQ-DM [8], and Temporal Feature Matters [10] further exploit temporal information to reduce quantization error across denoising steps. For DiT-based image generators, PTQ4DiT [28] and Q-DiT [2] study activation outliers and dynamic activation quantization in transformer diffusion backbones, while SVDQuant [14] isolates difficult outlier components with a low-rank branch. Beyond PTQ, QAT methods such as Q-DM [17] and EfficientDM [5] finetune quantized diffusion models with teacher distillation, improving low-bit fidelity compared with calibration-only approaches. More recent image-diffusion quantization frameworks also explore extremely low-bit weights and timestep-related feature handling [23, 15].

From image diffusion quantization to video diffusion quantization. Although these imagediffusion methods establish strong baselines, directly transferring them to VDMs is non-trivial. Video DiTs process much longer token sequences, making online transformations, rotation-based smoothing, or auxiliary compensation modules more costly at inference time. More importantly, video generation is sensitive not only to single-frame perceptual quality but also to temporal consistency, motion structure, and late-stage detail refinement. Existing VDM quantization methods mainly follow PTQ-style designs. QVD [24] targets video-specific activation distributions by observing skewed temporal features and inter-channel range disparities, then introduces high temporal discriminability quantization and scattered channel range integration to preserve temporal guidance and improve quantization-level coverage. ViDiT-Q [33] extends PTQ to image and video diffusion transformers by analyzing DiT-specific quantization errors and designing a DiT-oriented quantization scheme; its mixed-precision variant further allocates precision to sensitive layers and timesteps. S<sup>2</sup>Q-VDiT [4] focuses on the long-token calibration difficulty of video DiTs, using Hessian-aware salient data selection and attention-guided sparse token distillation to select more informative calibration samples and emphasize influential tokens. QVGen [9] introduces QAT to VDMs by adapting the teacherdistillation pipeline used in image diffusion quantization. In contrast, our work shows that this timestep-agnostic QAT objective is mismatched with the stage-wise behavior of video denoising: early steps benefit from teacher anchoring, while late steps require target-oriented detail refinement and reduced CFG-induced quantization-noise amplification.

## B Gradient Variance Analysis of Early-Stage Distillation

This section analyzes the variance-reduction aspect of early-stage distillation. The trajectoryanchoring effect follows from the KD objective itself, which explicitly penalizes deviations between the quantized student and the full-precision teacher.

We use the notation in Sec. 4.1. For a fixed noisy latent, condition, and timestep, let

$$
\begin{array} { r } { z = ( x _ { t } , c , t ) , \qquad s _ { Q } ( z ) = f ^ { q } ( x _ { t } , c , t ) , \qquad J _ { Q } ( z ) = \nabla _ { \theta _ { Q } } f ^ { q } ( x _ { t } , c , t ) , } \end{array}\tag{12}
$$

where $J _ { Q }$ denotes the surrogate Jacobian used by the Straight-Through Estimator (STE). Under squared-error training, the optimal predictor for the native diffusion or flow target is the conditional mean

$$
m ( z ) = \mathbb { E } [ y _ { t } \mid z ] .\tag{13}
$$

We decompose the sample-wise target as

$$
y _ { t } = m ( z ) + \xi _ { t } , \qquad \mathbb { E } [ \xi _ { t } \mid z ] = 0 , \qquad \operatorname { C o v } ( \xi _ { t } \mid z ) = \Sigma _ { t } ( z ) .\tag{14}
$$

Then the conditional expected target loss satisfies

$$
\begin{array} { r } { \mathbb { E } \Big [ \| s _ { Q } ( z ) - y _ { t } \| _ { 2 } ^ { 2 } \mid z \Big ] = \| s _ { Q } ( z ) - m ( z ) \| _ { 2 } ^ { 2 } + \operatorname { T r } \Sigma _ { t } ( z ) . } \end{array}\tag{15}
$$

Thus, the sample-wise target loss has the correct population optimum, but its stochastic gradient contains posterior target noise. Specifically, the sample gradient of the target loss in Eq. 3 is

$$
\begin{array} { r } { g _ { \mathrm { T a r g e t } } ( z , y _ { t } ) = \nabla _ { \theta _ { Q } } \mathcal { L } _ { \mathrm { T a r g e t } } ( t ) = 2 J _ { Q } ( z ) ^ { \top } \left( s _ { Q } ( z ) - y _ { t } \right) . } \end{array}\tag{16}
$$

and therefore

$$
\begin{array} { r } { \operatorname { C o v } ( g _ { \mathrm { T a r g e t } } \mid z ) = 4 J _ { Q } ( z ) ^ { \top } \Sigma _ { t } ( z ) J _ { Q } ( z ) . } \end{array}\tag{17}
$$

In early denoising steps, $x _ { t }$ is dominated by noise and contains limited information about the clean video latent $x _ { 1 }$ . For flow, velocity, and data-related targets of the generic form $y _ { t } = a _ { t } x _ { 1 } + b _ { t } \epsilon$ many plausible clean endpoints remain compatible with the same noisy state. Consequently, the posterior covariance $\Sigma _ { t } ( z )$ can be large, and Eq. (17) shows that this uncertainty is directly injected into the STE gradient. Such high-variance updates are especially harmful in low-bit QAT, where small gradient fluctuations can frequently move latent weights across quantization bin boundaries.

Distillation replaces the sample-wise target with the full-precision teacher prediction. A well-trained teacher can be written as

$$
\boldsymbol { f } ( \boldsymbol { x } _ { t } , c , t ) = \boldsymbol { m } ( z ) + \delta ( z ) ,\tag{18}
$$

where $\delta ( z )$ denotes the teacher residual with respect to the conditional mean. The sample gradient of the distillation loss in Eq. 4 is

$$
\begin{array} { r } { g _ { \mathrm { K D } } ( z ) = \nabla _ { \theta _ { Q } } \mathcal { L } _ { \mathrm { K D } } ( t ) = 2 J _ { Q } ( z ) ^ { \top } \left( s _ { Q } ( z ) - f ( x _ { t } , c , t ) \right) . } \end{array}\tag{19}
$$

Conditioned on the same $z ,$ the teacher output is deterministic, so the label-induced conditional covariance is removed:

$$
\mathrm { C o v } ( g _ { \mathrm { K D } } \mid z ) = 0 .\tag{20}
$$

Distillation therefore trades the variance term induced by $\xi _ { t }$ for the teacher residual $\delta ( z )$ . In early high-noise steps, the posterior target variance is large, while the teacher provides a stable estimate of the conditional denoising direction. KD is therefore preferable as a variance-reduction mechanism: it stabilizes QAT updates and anchors the global denoising trajectory.

This argument also explains why distillation should not dominate all timesteps. As denoising approaches the low-noise refinement regime, the variance-reduction benefit of KD becomes less dominant, while the teacher residual $\delta ( z )$ and the limited capacity of the quantized hypothesis space become more important. In this regime, optimizing the native target loss better aligns the quantized model with detail reconstruction instead of forcing it to imitate the full-precision teacher residual. This motivates our denoising-stage schedule, which uses KD in early steps and gradually shifts supervision toward the target loss in later steps.

## C Two-Stage Analysis of Late-Step CFG Quantization Error

We analyze late-step quantization degradation as a two-stage process: a mismatch first forms between the conditional and unconditional branch errors, and CFG then amplifies this mismatch. To isolate quantization effects from denoising-trajectory accumulation, we compare FP and quantized predictions under the same latent state, prompt, and timestep. Define

$$
e _ { c } ( t ) = f _ { c } ^ { q } ( t ) - f _ { c } ( t ) , \qquad e _ { \emptyset } ( t ) = f _ { \emptyset } ^ { q } ( t ) - f _ { \emptyset } ( t ) ,\tag{21}
$$

where $f _ { c } , f _ { \varnothing }$ and $f _ { c } ^ { q } , f _ { \varnothing } ^ { q }$ are the FP and quantized conditional and unconditional predictions, respectively.

Stage I: Formation of branch mismatch. We first explain why the two branch errors need not cancel under dynamic activation quantization. Consider a dynamic quantizer

$$
Q ( h ) = s ( h ) \operatorname { r o u n d } \left( { \frac { h } { s ( h ) } } \right) , \qquad e ( h ) = Q ( h ) - h ,\tag{22}
$$

where the scale $s ( h )$ is computed from the current activation tensor, token, or channel group. Although the two CFG branches share the same weights and noisy latent, they receive different text conditions. As these signals propagate through cross-attention, normalization, and feed-forward layers, they induce different activation distributions:

$$
h _ { c } ^ { \ell } \neq h _ { \varnothing } ^ { \ell } , \qquad s _ { c } ^ { \ell } = s ( h _ { c } ^ { \ell } ) , \qquad s _ { \varnothing } ^ { \ell } = s ( h _ { \varnothing } ^ { \ell } ) .\tag{23}
$$

In general, $s _ { c } ^ { \ell } \neq s _ { \varnothing } ^ { \ell } .$ , so the branches are rounded on different quantization grids. Their layer-wise errors are therefore not perfectly aligned, and the accumulated output-level errors $e _ { c } ( t )$ and $e _ { \emptyset } ( t )$ need not cancel. We characterize their alignment by

$$
\rho _ { t } = \frac { \mathbb { E } [ \langle e _ { c } ( t ) , e _ { \varnothing } ( t ) \rangle ] } { \sqrt { \mathbb { E } \| e _ { c } ( t ) \| _ { 2 } ^ { 2 } \mathbb { E } \| e _ { \varnothing } ( t ) \| _ { 2 } ^ { 2 } } } .\tag{24}
$$

Table 8: Identical CFG-drop controls on Wan2.1-1.3B. We compare each model with and without the same late-step CFG-drop schedule. QVGen and Base use symmetric W4A4 quantization; the full-precision model uses BF16.
<table><tr><td rowspan="2">Method</td><td rowspan="2">#Bits (W/A)</td><td colspan="3">VBench Avg↑</td><td colspan="3">LV↑</td></tr><tr><td>w/o drop</td><td>w/ drop</td><td> $\Delta$ </td><td>w/o drop</td><td>w/ drop</td><td> $\Delta$ </td></tr><tr><td>Full Prec.</td><td>16/16</td><td>66.28</td><td>66.44</td><td>+0.16</td><td>808</td><td>850</td><td>+42</td></tr><tr><td>QVGen Base</td><td>4/4 4/4</td><td>63.50 65.09</td><td>64.73 66.31</td><td>+1.23 +1.22</td><td>665 674</td><td>856 842</td><td>+191 +168</td></tr></table>

This formulation does not assume independence: the branches may share correlated error because they use the same weights and latent input. The relevant quantity is the residual mismatch after any correlated components cancel.

Let

$$
\sigma _ { c } ^ { 2 } ( t ) = \mathbb { E } \| e _ { c } ( t ) \| _ { 2 } ^ { 2 } , \qquad \sigma _ { \emptyset } ^ { 2 } ( t ) = \mathbb { E } \| e _ { \emptyset } ( t ) \| _ { 2 } ^ { 2 } ,\tag{25}
$$

and define the branch mismatch as

$$
\Delta ^ { q } ( t ) = e _ { c } ( t ) - e _ { \emptyset } ( t ) .\tag{26}
$$

Its expected energy is

$$
\begin{array} { r } { \mathbb { E } \| \Delta ^ { q } ( t ) \| _ { 2 } ^ { 2 } = \sigma _ { c } ^ { 2 } ( t ) + \sigma _ { \emptyset } ^ { 2 } ( t ) - 2 \rho _ { t } \sigma _ { c } ( t ) \sigma _ { \emptyset } ( t ) . } \end{array}\tag{27}
$$

Equation (27) separates two causes of a larger late-step mismatch. First, the individual branch errors can increase as increasingly structured late-stage activations become more sensitive to low-bit quantization. Second, weaker cross-branch error correlation reduces cancellation between the two errors. Together, these effects enlarge $\Delta ^ { q } ( t )$ before CFG is applied.

Stage II: Amplification of branch mismatch by CFG. The FP and quantized guided predictions under the same state are

$$
\hat { f } _ { \omega } ( t ) = f _ { \theta } ( t ) + \omega \big ( f _ { c } ( t ) - f _ { \theta } ( t ) \big ) , \qquad \hat { f } _ { \omega } ^ { q } ( t ) = f _ { \theta } ^ { q } ( t ) + \omega \big ( f _ { c } ^ { q } ( t ) - f _ { \theta } ^ { q } ( t ) \big ) .\tag{28}
$$

Their difference gives the final guided quantization error:

$$
\delta _ { \omega } ^ { q } ( t ) = \hat { f } _ { \omega } ^ { q } ( t ) - \hat { f } _ { \omega } ( t ) = \underbrace { e _ { \theta } ( t ) } _ { \mathrm { s i n g l e - b r a n c h ~ e r r o r } } + \omega \underbrace { \vphantom { e _ { \theta } ^ { q } ( t ) } \left( e _ { c } ( t ) - e _ { \theta } ( t ) \right) } _ { \mathrm { b r a n c h ~ m i s m a t c h ~ } \Delta ^ { q } ( t ) } .\tag{29}
$$

Equation (29) makes the second stage explicit: after the branch mismatch $\Delta ^ { q } ( t )$ has formed in Stage I, CFG directly scales its contribution to the final guided error by ω.

This amplification is most harmful in late denoising steps. In this regime, the useful FP guidance direction

$$
f _ { c } ( t ) - f _ { \varnothing } ( t )\tag{30}
$$

becomes relatively weak because the latent already contains rich visual structure, while the branch mismatch formed in Stage I can continue to grow. The amplified term $\omega \Delta ^ { q } ( t )$ can therefore dominate the useful guidance signal and appear as high-frequency artifacts. Denoising-Stage Gated Guidance removes this multiplier in the final steps by using only the conditional prediction, suppressing Stage II amplification after the useful FP guidance direction has diminished.

## D General and Quantization-Specific Effects of CFG-Drop

Late-step CFG suppression can also benefit full-precision diffusion sampling, raising the question of whether CFG-drop is merely a general sampling heuristic. To separate this generic effect from its role in quantized models, we apply an identical CFG-drop schedule to the full-precision model, QVGen, and our engineering-only quantized baseline (Base), while keeping all other inference settings fixed. Table 8 reports both the VBench average and Laplacian Variance (LV), where LV measures visual sharpness.

CFG-drop yields a small improvement for the full-precision model, so we do not attribute its entire effect exclusively to quantization. However, both quantized models receive substantially larger gains in VBench average and, especially, visual sharpness. These controlled results show that the improvement cannot be explained solely by a generic sampling benefit. Instead, CFG-drop is particularly effective under low-bit quantization, consistent with the mechanism in Appendix C: once the late-stage branch mismatch has formed, removing CFG prevents this quantization-specific error from being further amplified.

## E Detailed Training Settings

The detailed training settings including batch size, steps, training frames, and learning rate are provided in Tab. 9. We train our model on H20 GPUs with 140GB memory. By default we use local batch=1 for training, so the global batch size is equal to the number of GPUs we use.

For training data, we use synthetic samples. For Wan2.1-1.3B, we synthesize data from prompts in the filtered and LLM-extended version of VidProM [26], following previous works [7]. For Wan2.1-14B, we directly use the synthetic latent data released by TurboDiffusion [32]. For the other models, to reduce data synthesis cost, we decode the TurboDiffusion latents into videos with the Wan2.1 VAE and then re-encode the videos with the corresponding model-specific VAE to obtain latent training data.

Table 9: Detailed training settings for each model. We set bz=1 for each gpu to evaluate training time.
<table><tr><td>Models</td><td>Batch size</td><td>Total steps</td><td>Warmup steps</td><td> $\mathrm { L r }$ </td><td>Frames</td><td>Resolution</td><td>Time (H20 days)</td></tr><tr><td>CogVideoX-2B</td><td>32</td><td>2000</td><td>200</td><td>3e-5</td><td>49</td><td>480p</td><td>6.07</td></tr><tr><td>CogVideoX1.5-5B</td><td>32</td><td>2000</td><td>200</td><td>5e-5</td><td>81</td><td>720p</td><td>~65</td></tr><tr><td>Wan2.1 1.3B</td><td>24</td><td>2000</td><td>200</td><td>3e-5</td><td>81</td><td>480p</td><td>8.19</td></tr><tr><td>Wan2.1 14B</td><td>64</td><td>1000</td><td>100</td><td>3e-5</td><td>81</td><td>480p</td><td>~64</td></tr><tr><td>Wan2.2 5B</td><td>32</td><td>4000</td><td>400</td><td>3e-5</td><td>81</td><td>480p</td><td>6.9</td></tr></table>

## F Detailed Evaluation Settings

Following QVGen [9], we set the random seed according to the process rank at the beginning of evaluation, ensuring that each generated video is sampled from different random noise. We use 128 GPUs for inference on Wan2.1-14B and 32 GPUs for all other models.

## G Timestep-Conditioned Linear Layers and LUT Precomputation

Previous quantization research [23, 15, 8] on image diffusion models has demonstrated that precomputing time features during inference can enhance accuracy without compromising inference efficiency. This is feasible because, during the inference phase, timesteps are discrete integers (e.g., 1,000 distinct values ranging from 0 to 999), forming a finite set. Therefore, we can pre-compute the time features for all timesteps and store them on the CPU. During inference, only the specific feature corresponding to the current timestep is dynamically loaded onto the GPU. Consequently, all timestep-related linear layers can be entirely removed without incurring any additional GPU memory overhead. Below, we detail the procedure for precomputing the Look-Up Table (LUT) for CogVideoX and Wan, alongside a size comparison between the generated LUT and the original linear layers.

We analyze the timestep-conditioned computation in CogVideoX-2B and Wan2.1-T2V-1.3B. Let the inference scheduler use a fixed set of denoising timesteps

$$
\mathcal { T } _ { K } = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { K } \} ,\tag{31}
$$

where we use $K = 5 0$ in the main comparison. Unless otherwise specified, all storage costs are computed with fp16/bf16 precision, i.e., 2 bytes per scalar.

## G.1 CogVideoX-2B

For CogVideoX-2B, the hidden dimension and timestep embedding dimension are

$$
D _ { c } = 3 0 \times 6 4 = 1 9 2 0 , \qquad E _ { c } = 5 1 2 , \qquad L _ { c } = 3 0 .\tag{32}
$$

Given a scheduler timestep t, CogVideoX first computes a sinusoidal timestep embedding and then maps it through a timestep MLP:

$$
\begin{array} { r } { p _ { t } = \mathrm { S i n E m b } _ { D _ { c } } ( t ) \in \mathbb { R } ^ { D _ { c } } , } \end{array}\tag{33}
$$

$$
h _ { t } = \mathrm { T E m b } ( p _ { t } ) \in \mathbb { R } ^ { E _ { c } } .\tag{34}
$$

Here SinEmb $\smash { D _ { c } \mathopen { } \mathclose \bgroup \left( \cdot \aftergroup \egroup \right) }$ is deterministic and has no trainable linear weight. The trainable timestep MLP can be written as

$$
h _ { t } = W _ { t } ^ { ( 2 ) } \sigma \left( W _ { t } ^ { ( 1 ) } p _ { t } + b _ { t } ^ { ( 1 ) } \right) + b _ { t } ^ { ( 2 ) } ,\tag{35}
$$

where $\sigma ( \cdot )$ denotes the activation function.

Each CogVideoX transformer block contains two timestep-conditioned AdaLN-Zero modules, one before attention and one before the feed-forward network. For block $\ell \in \{ 1 , \ldots , L _ { c } \}$ and AdaLN index $r \in \{ 1 , 2 \}$ , the timestep-conditioned modulation vector is

$$
a _ { \ell , r } ( t ) = W _ { \ell , r } \sigma ( h _ { t } ) + b _ { \ell , r } \in \mathbb { R } ^ { 6 D _ { c } } .\tag{36}
$$

This vector is split into six per-channel vectors:

$$
a _ { \ell , r } ( t ) = \left[ \beta _ { \ell , r } ^ { v } ( t ) , \gamma _ { \ell , r } ^ { v } ( t ) , g _ { \ell , r } ^ { v } ( t ) , \beta _ { \ell , r } ^ { e } ( t ) , \gamma _ { \ell , r } ^ { e } ( t ) , g _ { \ell , r } ^ { e } ( t ) \right] ,\tag{37}
$$

where

$$
\begin{array} { r l r } & { \beta _ { \ell , r } ^ { v } , \gamma _ { \ell , r } ^ { v } , g _ { \ell , r } ^ { v } \in \mathbb { R } ^ { D _ { c } } , \quad } & { \beta _ { \ell , r } ^ { e } , \gamma _ { \ell , r } ^ { e } , g _ { \ell , r } ^ { e } \in \mathbb { R } ^ { D _ { c } } . } \end{array}\tag{38}
$$

The final output AdaLN produces only shift and scale:

$$
o ( t ) = W _ { o } \sigma ( h _ { t } ) + b _ { o } \in \mathbb { R } ^ { 2 D _ { c } } ,\tag{39}
$$

$$
o ( t ) = [ \beta _ { o } ( t ) , \gamma _ { o } ( t ) ] , \qquad \beta _ { o } ( t ) , \gamma _ { o } ( t ) \in \mathbb { R } ^ { D _ { c } } .\tag{40}
$$

Therefore, for a fixed inference timestep set $\mathcal { T } _ { K } .$ , the CogVideoX timestep LUT can be defined as

$$
\mathcal { L } _ { \mathrm { C o g } } = \left\{ a _ { \ell , r } ( t _ { s } ) , o ( t _ { s } ) ~ | ~ t _ { s } \in \mathcal { T } _ { K } , \ell = 1 , \ldots , L _ { c } , r \in \left\{ 1 , 2 \right\} \right\} .\tag{41}
$$

The number of cached scalar values is

$$
\left| \mathcal { L } _ { \mathrm { C o g } } \right| = K \left( 2 L _ { c } \cdot 6 D _ { c } + 2 D _ { c } \right) .\tag{42}
$$

Substituting $K = 5 0 , L _ { c } = 3 0$ , and $D _ { c } = 1 9 2 0$ gives

$$
| { \mathcal { L } } _ { \mathrm { C o g } } | = 5 0 \left( 2 \cdot 3 0 \cdot 6 \cdot 1 9 2 0 + 2 \cdot 1 9 2 0 \right) = 3 4 { , } 7 5 2 { , } 0 0 0 .\tag{43}
$$

Thus, the fp16/bf16 LUT size is

$$
3 4 , 7 5 2 , 0 0 0 \times 2 = 6 9 , 5 0 4 , 0 0 0 { \mathrm { ~ b y t e s } } \approx 6 9 . 5 0 { \mathrm { ~ M B } } \approx 6 6 . 2 8 { \mathrm { ~ M i B } } .\tag{44}
$$

The total number of timestep-related linear parameters in CogVideoX-2B is

$$
\begin{array} { r l } & { N _ { \mathrm { C o g } } = \underbrace { \left( D _ { c } E _ { c } + E _ { c } \right) + \left( E _ { c } ^ { 2 } + E _ { c } \right) } _ { \mathrm { t i m e ~ e m b e d d i n g ~ M L P } } } \\ & { \quad \quad + \underbrace { 2 L _ { c } \left( ( 6 D _ { c } ) E _ { c } + 6 D _ { c } \right) } _ { \mathrm { b l o c k ~ A d a L N l i n e a r s } } } \\ & { \quad \quad + \underbrace { \left( 2 D _ { c } \right) E _ { c } + 2 D _ { c } } _ { \mathrm { f i n a l ~ A d a L N l i n e a r } } . } \end{array}\tag{45}
$$

Substituting $D _ { c } = 1 9 2 0 , E _ { c } = 5 1 2$ , and $L _ { c } = 3 0$

$$
N _ { \mathrm { C o g } } = 3 5 7 , 8 0 1 , 7 2 8 .\tag{46}
$$

Therefore, the fp16/bf16 parameter storage is

$$
3 5 7 , 8 0 1 , 7 2 8 \times 2 = 7 1 5 , 6 0 3 , 4 5 6 { \mathrm { ~ b y t e s } } \approx 7 1 5 . 6 0 { \mathrm { ~ M B } } \approx 6 8 2 . 4 5 { \mathrm { ~ M i B } } .\tag{47}
$$

## G.2 Wan2.1-T2V-1.3B

For Wan2.1-T2V-1.3B, the hidden dimension, sinusoidal frequency dimension, and number of transformer layers are

$$
D _ { w } = 1 5 3 6 , \qquad F _ { w } = 2 5 6 , \qquad L _ { w } = 3 0 .\tag{48}
$$

Given timestep t, Wan first computes a sinusoidal embedding:

$$
r _ { t } = \mathrm { S i n E m b } _ { F _ { w } } ( t ) \in \mathbb { R } ^ { F _ { w } } .\tag{49}
$$

Then the global timestep embedding is computed as

$$
\begin{array} { r } { e _ { t } = W _ { e } ^ { ( 2 ) } \sigma \left( W _ { e } ^ { ( 1 ) } r _ { t } + b _ { e } ^ { ( 1 ) } \right) + b _ { e } ^ { ( 2 ) } \in \mathbb { R } ^ { D _ { w } } . } \end{array}\tag{50}
$$

Wan then computes a global six-way timestep projection:

$$
e _ { t } ^ { 0 } = \mathrm { r e s h a p e } \left( W _ { p } \sigma ( e _ { t } ) + b _ { p } \right) \in \mathbb { R } ^ { 6 \times D _ { w } } .\tag{51}
$$

Unlike CogVideoX-2B, Wan does not use a separate large timestep-conditioned linear layer inside each transformer block. Instead, each block ℓ has a learned modulation tensor

$$
M _ { \ell } \in \mathbb { R } ^ { 6 \times D _ { w } } .\tag{52}
$$

The block-specific timestep modulation is obtained by a lightweight element-wise addition:

$$
\begin{array} { r } { q _ { \ell } ( t ) = M _ { \ell } + e _ { t } ^ { 0 } \in \mathbb { R } ^ { 6 \times D _ { w } } . } \end{array}\tag{53}
$$

This tensor is split into six vectors:

$$
q _ { \ell } ( t ) = \left\lceil \beta _ { \ell } ^ { a } ( t ) , \gamma _ { \ell } ^ { a } ( t ) , g _ { \ell } ^ { a } ( t ) , \beta _ { \ell } ^ { f } ( t ) , \gamma _ { \ell } ^ { f } ( t ) , g _ { \ell } ^ { f } ( t ) \right\rceil ,\tag{54}
$$

where the superscripts a and $f$ denote the self-attention and feed-forward branches, respectively.

The final output head also uses the global timestep embedding $e _ { t }$ . Let the head modulation be

$$
M _ { \mathrm { h e a d } } \in \mathbb { R } ^ { 2 \times D _ { w } } .\tag{55}
$$

Then

$$
q _ { \mathrm { h e a d } } ( t ) = M _ { \mathrm { h e a d } } + e _ { t } \in \mathbb { R } ^ { 2 \times D _ { w } } ,\tag{56}
$$

which is split as

$$
q _ { \mathrm { h e a d } } ( t ) = [ \beta _ { \mathrm { h e a d } } ( t ) , \gamma _ { \mathrm { h e a d } } ( t ) ] .\tag{57}
$$

Therefore, for Wan2.1-T2V-1.3B, it is sufficient to cache only the global timestep quantities:

$$
\mathcal { L } _ { \mathrm { W a n } } = \left\{ e _ { t _ { s } } , e _ { t _ { s } } ^ { 0 } \ \middle | \ t _ { s } \in \mathcal { T } _ { K } \right\} .\tag{58}
$$

The number of cached scalar values is

$$
\left| \mathcal { L } _ { \mathrm { W a n } } \right| = K \left( D _ { w } + 6 D _ { w } \right) = 7 K D _ { w } .\tag{59}
$$

Substituting K = 50 and $D _ { w } = 1 5 3 6$ gives

$$
| { \mathcal { L } } _ { \mathrm { W a n } } | = 5 0 \times 7 \times 1 5 3 6 = 5 3 7 { \mathrm { , 6 0 0 . } }\tag{60}
$$

Thus, the fp16/bf16 LUT size is

$$
{ 5 3 7 , 6 0 0 \times 2 } = 1 { , } 0 7 5 { , } 2 0 0 \mathrm { ~ b y t e s } \approx 1 { . } 0 7 5 \mathrm { M B } \approx 1 { . } 0 2 5 \mathrm { M i B } { . }\tag{61}
$$

The total number of timestep-related linear parameters in Wan2.1-T2V-1.3B is

$$
\begin{array} { r l } & { N _ { \mathrm { W a n } } = \underbrace { \left( F _ { w } D _ { w } + D _ { w } \right) + \left( D _ { w } ^ { 2 } + D _ { w } \right) } _ { \mathrm { t i m e ~ e m b e d d i n g ~ M L P } } } \\ & { ~ + \underbrace { \left( 6 D _ { w } \right) D _ { w } } _ { \mathrm { g l o b a l ~ t i m e ~ p r o j e c t i o n } } . } \end{array}\tag{62}
$$

Substituting $F _ { w } = 2 5 6$ and $D _ { w } = 1 5 3 6 \mathrm { : }$

$$
N _ { \mathrm { W a n } } = 1 6 , 9 2 0 , 5 7 6 .\tag{63}
$$

Therefore, the fp16/bf16 parameter storage is

$$
1 6 , 9 2 0 , 5 7 6 \times 2 = 3 3 , 8 4 1 , 1 5 2 \mathrm { b y t e s } \approx 3 3 . 8 4 \mathrm { M B } \approx 3 2 . 2 7 \mathrm { M i B } .\tag{64}
$$

Table 10: Comparison of timestep-related linear parameters and 50-step LUT size.
<table><tr><td>Model</td><td>Time-related linear scope</td><td>Params</td><td>Weight size</td><td>50-step LUT content</td><td>LUT size</td></tr><tr><td>CogVideoX-2B</td><td>time embedding MLP + block AdaLN linears + final AdaLN linear</td><td>357,801,728</td><td>715.60 MB / 682.45 MiB</td><td>all block AdaLN parameters + final AdaLN parameters</td><td>69.50 MB / 66.28 MiB</td></tr><tr><td>Wan2.1-T2V-1.3B</td><td>time embedding MLP + global time projection</td><td>16,920,576</td><td>33.84 MB / 32.27 MiB</td><td> $\begin{array} { r } { \mathbf { g l o b a l } ~ e _ { t } \in \mathbb { R } ^ { 1 5 3 6 } } \\ { \mathbf { a n d } ~ e _ { t } ^ { 0 } \in \mathbb { R } ^ { 6 \times 1 5 3 6 } } \end{array}$ </td><td>1.075 MB / 1.025 MiB</td></tr></table>

Table 11: Supplementary table for VBench results in Tab. 1.
<table><tr><td rowspan="2">Method</td><td rowspan="2">#Bits (W/A)</td><td colspan="8">VBench1↑</td></tr><tr><td>Imaging Quality</td><td>Aesthetic Quality</td><td>Motion Smoothness</td><td>Dynamic Degree</td><td>Background Consistency</td><td>Subject Consistency</td><td>Scene Consistency</td><td>Overall Consistency</td><td> $\mathbf { A v } \mathbf { g }$ </td></tr><tr><td colspan="10">Wan2.1 1.3B (CFG = 5.0, 480p, fps = 16)</td></tr><tr><td>FP</td><td>16/16</td><td>64.22</td><td>57.85</td><td>97.25</td><td>71.11</td><td>95.72</td><td>93.35</td><td>26.07</td><td>24.70</td><td>66.28</td></tr><tr><td>QVGen</td><td>3/3</td><td>59.48</td><td>50.33</td><td>97.91</td><td>42.50</td><td>94.78</td><td>89.96</td><td>11.68</td><td>20.32</td><td>58.37</td></tr><tr><td>Ours</td><td>3/3</td><td>61.90</td><td>54.56</td><td>96.41</td><td>77.77</td><td>94.46</td><td>90.88</td><td>22.46</td><td>24.11</td><td>65.32</td></tr><tr><td colspan="9">Wan2.2 5B (CFG = 5.0, 720p, fps = 16)</td></tr><tr><td>QVGen</td><td>3/3</td><td>60.65</td><td>54.95</td><td>97.97</td><td>55.0</td><td>95.08</td><td>92.43</td><td>19.56</td><td>23.49</td><td>62.39</td></tr><tr><td>Ours</td><td>3/3</td><td>61.22</td><td>58.53</td><td>98.50</td><td>60.00</td><td>95.74</td><td>94.92</td><td>24.26</td><td>25.27</td><td>64.81</td></tr></table>

## G.3 Why LUT Precomputation is Feasible

For both models, the cached quantities are deterministic functions of the discrete scheduler timestep and fixed model weights:

$$
\mathcal { L } ( t ) = f _ { \theta } ( t ) , \qquad t \in \mathcal { T } _ { K } ,\tag{65}
$$

where θ denotes the frozen parameters of the timestep-conditioned linear layers.

Importantly, these quantities do not depend on the latent tokens, text tokens, attention activations, or feed-forward activations:

$$
\frac { \partial \mathcal { L } ( t ) } { \partial X } = 0 , ~ \frac { \partial \mathcal { L } ( t ) } { \partial C } = 0 .\tag{66}
$$

Thus, given a fixed inference scheduler, the timestep-conditioned linear outputs can be precomputed once before denoising:

$$
\left\{ f _ { \theta } ( t _ { 1 } ) , f _ { \theta } ( t _ { 2 } ) , \ldots , f _ { \theta } ( t _ { K } ) \right\} ,\tag{67}
$$

and reused throughout the denoising process by indexing the LUT with the current timestep.

This removes online execution of the timestep-related linear layers while preserving the original LayerNorm, affine modulation, attention, feed-forward, and residual-gating computation paths.

## G.4 Storage Comparison

The total storage cost comparison is summarized in Tab. 10.

## H Supplementary Results for Main Experiments

This section provides supplementary VBench results, summarized in Tab. 11, for the main comparison in Tab. 1. We focus on the more aggressive W3A3 setting for Wan2.1-1.3B and Wan2.2-5B, complementing the main-table results with per-dimension scores under standard prompts.

## I Full VBench Results For Ablation Studies

Due to page constraints in the main text, the ablation study section only reports five sensitive metrics from VBench. The comprehensive evaluation results across all metrics are summarized in Tabs. 12 and 13.

Table 12: Supplementary table for ablation study in Tab. 4.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Mixed loss</td><td rowspan="2">CFG drop</td><td rowspan="2">#Bits (W/A)</td><td colspan="9">VBench1↑</td></tr><tr><td>Aesthetic Motion Quality</td><td>Smoothness</td><td>Dynamic Degree</td><td>Background Consistency</td><td>Subject Consistency</td><td></td><td>Scene Consistency</td><td>Overall Consistency</td><td>Avg</td></tr><tr><td></td><td></td><td></td><td colspan="10">Wan 1.3B (CFG = 5.0, 480p, fps = 16)</td></tr><tr><td>FP</td><td>一</td><td>一</td><td>16/16</td><td>64.22</td><td>57.85</td><td>97.25</td><td>71.11</td><td>95.72</td><td>93.35</td><td>26.07</td><td>24.70</td><td>66.28</td></tr><tr><td>Base</td><td>X</td><td>X</td><td>4/4</td><td>62.86</td><td>56.32</td><td>96.23</td><td>71.67</td><td>94.68</td><td>91.69</td><td>22.94</td><td>24.35</td><td>65.09</td></tr><tr><td>Base + CFG-drop</td><td>X</td><td>√</td><td>4/4</td><td>61.67</td><td>56.75</td><td>96.57</td><td>77.50</td><td>96.57</td><td>92.57</td><td>25.58</td><td>24.20</td><td>66.31</td></tr><tr><td>Base + Mixed loss</td><td>√</td><td>X</td><td>4/4</td><td>64.06</td><td>57.47</td><td>96.07</td><td>76.67</td><td>94.45</td><td>91.52</td><td>25.16</td><td>24.55</td><td>66.24</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>4/4</td><td>63.52</td><td>58.01</td><td>96.67</td><td>80.00</td><td>95.52</td><td>92.96</td><td>26.54</td><td>24.49</td><td>67.20</td></tr></table>

Table 13: Supplementary table for ablation study in Tab. 5.

<table><tr><td rowspan="2">Method</td><td rowspan="2">#Bits (W/A)</td><td colspan="9">VBench1↑</td></tr><tr><td>Imaging Quality</td><td>Aesthetic Quality</td><td>Motion Smoothness</td><td>Dynamic Degree</td><td>Background Consistency</td><td>Subject Consistency</td><td>Scene Consistency</td><td>Overall Consistency</td><td>Avg</td></tr><tr><td colspan="10">Wan 1.3B (CFG = 5.0, 480p, fps = 16)</td></tr><tr><td>FP</td><td>16/16</td><td>64.22</td><td>57.85</td><td>97.25</td><td>71.11</td><td>95.72</td><td>93.35</td><td>26.07</td><td>24.70</td><td>66.28</td></tr><tr><td>linear</td><td>4/4</td><td>62.8</td><td>57.1</td><td>97.80</td><td>70.55</td><td>95.88</td><td>93.73</td><td>20.22</td><td>23.84</td><td>65.24</td></tr><tr><td>exp1</td><td>4/4</td><td>63.42</td><td>57.03</td><td>97.52</td><td>73.61</td><td>95.71</td><td>93.79</td><td>22.60</td><td>23.85</td><td>65.94</td></tr><tr><td>exp5</td><td>4/4</td><td>63.53</td><td>58.01</td><td>96.67</td><td>80.00</td><td>95.52</td><td>92.96</td><td>26.54</td><td>24.49</td><td>67.20</td></tr><tr><td>exp10</td><td>4/4</td><td>63.50</td><td>57.71</td><td>97.70</td><td>74.72</td><td>95.77</td><td>93.49</td><td>18.79</td><td>23.43</td><td>65.64</td></tr></table>

## J Further inference acceleration

We further integrate SageAttention to accelerate inference. The results are summarized in Tab. 14.

Table 14: Inference latency when combining W4A4 quantization with SageAttention. All results are measured under 81 frames at 480p.
<table><tr><td>Model</td><td>Methods</td><td>4090D latency(s)</td></tr><tr><td rowspan="2">Wan 2.1-1.3B</td><td>Ours</td><td>193</td></tr><tr><td>Ours + SageAttention</td><td>140</td></tr><tr><td rowspan="2">Wan 2.1-14B</td><td>Ours</td><td>930</td></tr><tr><td>Ours + SageAttention</td><td>698</td></tr><tr><td rowspan="2">Wan 2.2-5B</td><td>Ours</td><td>58</td></tr><tr><td>Ours + SageAttention</td><td>53</td></tr></table>

## K Visual Comparison

We provide representative qualitative comparisons across three Wan family backbones: Wan2.1- 1.3B, Wan2.1-14B, and Wan2.2-5B. All samples are generated with the same prompts and inference settings as the quantitative evaluation, allowing a direct comparison between DSAQuant and prior QAT baselines. These examples complement the numerical results by highlighting differences in text-video alignment, texture fidelity, local detail preservation, and quantization-induced artifacts. Specifically, Figs. 5 and 6 show W4A4 and W3A3 results on Wan2.1-1.3B, Figs. 7 and 8 show results on Wan2.2-5B, and Figs. 9 and 10 show results on Wan2.1-14B. Full video comparisons are provided in the supplementary material.

## L Limitations, Broader impacts

Limitations. Our experiments are currently conducted on bidirectional video diffusion models. Recent advances in autoregressive video generation show strong potential for real-time generation, and evaluating DSAQuant on such models is an important direction for future work.

![](images/7d183340ac5f0aff1b55ecf8b7ee073095449d5215acfcba9d658dd7019fc84a.jpg)  
Figure 5: 4bit quantization on Wan2.1 1.3b model with prompt:"CG animation digital art, a majestic jellyfish floating gracefully through the oceanic depths. The jellyfish has intricate patterns on its translucent body, with vibrant hues of blue, green, and purple. Its bioluminescent tentacles emit a soft, mesmerizing glow, creating an ethereal underwater landscape. The tentacles sway gently as the jellyfish glides, illuminating the surrounding water with a captivating light show. The ocean background is filled with schools of colorful fish and drifting coral reefs. The jellyfish is surrounded by a serene, tranquil atmosphere. Soft, ambient ocean sounds play in the background. Low-angle, slow-motion shot focusing on the jellyfish and its glowing tentacles."

Broader impacts. This work improves the efficiency of video diffusion models through quantizationaware training. By reducing memory and computational costs while preserving visual fidelity, DSAQuant can make high-quality video generation more accessible and practical on resourceconstrained hardware. This may benefit creative production, education, assistive applications, and interactive media. However, efficient video generation also raises potential risks. Lowering the deployment cost of high-fidelity VDMs may make it easier to generate misleading synthetic videos, impersonation content, privacy-violating media, or other harmful outputs. Efficiency gains may also increase overall usage, partially offsetting the environmental benefits of reduced per-sample computation. We therefore encourage the deployment of such models together with safeguards such as watermarking, provenance tracking, content filtering, safety classifiers, and clear disclosure of synthetic content.

## M LLM Usage

In this paper, Large Language Models (LLMs) were used to assist with polishing the text and formatting the tables.

![](images/33c3f351abae8e23ede73552c82dd74e6f35b1f356a4edffc202ec3627312830.jpg)  
Figure 6: 3bit quantization on Wan2.1 1.3b model with prompt:"CG animation digital art, an African elephant taking a peaceful walk through a lush green forest at dawn. The elephant has a grayish-brown coat with soft folds and wrinkles, and gentle eyes. It walks slowly, with its trunk raised slightly, sniffing the air. The forest is filled with tall trees and vibrant greenery, with colorful flowers blooming. Birds fly overhead, chirping melodiously. The sky is a beautiful orange hue as the sun rises. The elephant moves gracefully, with each step careful and deliberate. The background features intricate foliage and subtle shadows. Soft lighting creates a warm and serene atmosphere. Low-angle shot from behind, focusing on the elephant’s tranquil expression and movement."

![](images/d4ef319f9124a9122a995de0d92b9570e8c7cbef4416fa648e613ec7344d2a72.jpg)  
Figure 7: 4bit quantization on Wan2.2 5b model with prompt:"CG animation digital art, a sleek racing bicycle speeding down a winding mountain road. The bike is a deep metallic silver color with intricate patterns etched along its frame. It accelerates rapidly, the rider gripping the handlebars tightly with focused determination. The rider has short, wavy brown hair and piercing blue eyes, wearing a tight-fitting black helmet and a snugly fitting silver jersey. They lean forward slightly, their arms pumping hard to maintain momentum. The road is rugged, covered in loose gravel and rocks, with trees lining the sides. The sun sets behind them, casting dramatic shadows. The background features a sunset sky with wispy clouds and hints of orange and pink hues. The bike’s wheels spin furiously as it gains speed, creating a whirlwind of dust and debris. The scene captures the adrenaline rush of a thrilling bicycle race. Close-up, low-angle view."

![](images/61b2e26461619bf7010db4bfae2262c2ccafa1de576894dbce4270759afd0d76.jpg)  
Figure 8: 3bit quantization on Wan2.2 5b model with prompt:"A playful feline sprinting joyfully across a lush green meadow dotted with wildflowers. The cat has sleek fur, expressive green eyes, and a fluffy tail that wags excitedly as it bounds forward. The meadow stretches out behind it, with vibrant sunflowers and buttercups swaying gently in the breeze. The sky above is a bright azure, filled with fluffy white clouds. The cat’s joyful run is captured from a dynamic low-angle perspective, showcasing its agility and boundless energy. The scene is bathed in warm golden light, enhancing the cat’s lively demeanor. Grass and petals trail behind the cat as it dashes towards the horizon. The background features a serene rural landscape, with small cottages and winding country roads visible in the distance. The overall composition is energetic and full of life, perfectly capturing the essence of a cat running happily."

![](images/e2937b5bb53fc4d0c3338be698dbd720853f90bd687dba43b9f8a456956253ba.jpg)  
Figure 9: 4bit quantization on Wan2.1 14b model with prompt:"CG animation concept art, a young woman with natural beauty applying makeup in the morning. She is wearing a simple white blouse and black pencil skirt. Her hair is styled in loose waves, framing her face. She is sitting at a vanity table with soft lighting illuminating her work. She is using a compact mirror to apply foundation, concealer, and blush. Her fingers move gracefully as she blends and applies products. She pauses occasionally to check her reflection and adjust her technique. The background features a minimalist room with a few scattered items and a vintage vanity set. Soft, gentle brush strokes and subtle motions. Low-angle shot from above, focusing on her hands and facial expressions."

![](images/329eeeccfe8bb8b0303d03184b9d5d6cbca4709211e3c4e4c2ec55a6d8834dac.jpg)  
Figure 10: 3bit quantization on Wan2.1 14b model with prompt:"A thrilling motorcycle speeding down a winding mountain road at night. The sleek black motorcycle with bright red accents accelerates rapidly, leaving behind a trail of dust. The rider, a muscular man with short cropped hair, wears a black leather jacket and jeans. His helmet is off, revealing a determined and focused expression. He grips the handlebars tightly, leaning slightly into the turn. The motorcycle’s engine roars as it gains momentum, reflecting the intense speed and power. The background is a dimly lit mountain landscape, with flickering streetlights and twinkling stars. The scene captures the adrenaline rush and raw energy of the motorcycle’s acceleration. Nighttime low-angle shot from the side."