# Conditional Flow Matching for Cross-Field MRI Harmonisation

Baris Imre<sup>1</sup> #, Aram Salehi<sup>2,3</sup>, Levente Baljer<sup>4</sup>, Andrew Webb<sup>3,5</sup>, Marius Staring<sup>1</sup>, Efe Ilicak<sup>1</sup>

<sup>1</sup> Division of Image Processing, Department of Radiology, Leiden University Medical Center

Department of Human Genetics, Radboud University Medical Center 3 Department of Radiology, Leiden University Medical Center 4 Department of Neuroimaging, King’s College London 5 Department of Bioengineering, University of Illinois

Abstract. Magnetic resonance images of the same subject look markedly diferent across field strengths, which complicates the comparison and pooling of data across sites. We address cross-field brain-MRI translation for the MRIxFields2026 challenge, and in particular its Task 3: a single model that translates between any directed pair of the five field strengths and across three contrasts. We phrase the problem as a conditional flow matching path: because the source and target volumes are spatially registered, we learn a velocity field that carries the source slice directly to the target slice, rather than starting from noise. To learn this mapping from only three paired subjects, the unified model is trained in three stages: a degradation-bridge pretraining that distills a restoration prior from the abundant unpaired retrospective cohort, a cross-field finetuning over all directed pairs on the paired cohort, and an adversarial refinement that sharpens the output. At inference, we integrate the learned velocity with a second-order Heun solver in a handful of steps. A restoration prior learned without any paired data already reaches a mean SSIM of 0.837, and each subsequent training stage improves on it. A single 6.3M-parameter model thereby covers all 60 field-pair and contrast combinations, with inference in five solver steps per slice. On the challenge evaluation set the model reaches a mean SSIM of 0.909, averaged over the three contrasts, outperforming regression and difusion baselines built on the identical network on all three challenge metrics.

Keywords: Conditional Flow Matching · MRI Field Translation · Generative Models

## 1 Introduction

Magnetic resonance imaging is central to neuroimaging. Typical MRI examinations consist of weighted images, as opposed to quantitative maps, that depend on tissue properties as well as acquisition settings. This makes images hard to compare across sites and hardware, hindering the reproducibility and the pooling of data for large studies.

![](images/a46f558d2153edd92f6be110654a61b6c7dfbc6bdb2e03b7a90d30d40f73c895.jpg)  
Fig. 1. The overall training architecture. Training is split into three stages, shown here as (1), (2), (3). In the pretraining stage the source (x<sub>0</sub>) is a degraded copy of the target (x<sub>1</sub>), since this cohort is unpaired; the network learns to restore it. In the second and third stages we continue with paired training. In the third stage, adversarial refinement, a discriminator judges the outputs of our CFM model and supplies an adversarial loss. The source field s, target field τ, and the timestep t are embedded and injected through FiLM. During training the network predicts a velocity that transports the source towards the target; at inference this velocity is integrated over several steps.

Field strength heavily contributes to the signal-to-noise ratio and the clinically achievable spatial resolution, and shifts tissue contrast. The same subject can look diferent at diferent fields, even for the same protocol. Translating a scan from one field to another would put these images on common ground.

The MRIxFields2026 challenge poses this as cross-field MRI translation across five field strengths and three contrasts. There are three tasks. Task 1 maps any field up to 7T, Task 2 maps 0.1T up to any higher field, and Task 3 demands a single model that translates between any directed pair of fields. In this paper we mainly focus on Task 3. We propose a conditional flow matching method that bridges the source slice to the target with a time-dependent velocity field, conditioned on the timestep and the source and target field strengths through FiLM, with the three contrasts stacked as input channels. The unified model is trained in three stages: a degradation-bridge pretraining that learns a restoration prior from unpaired data, a cross-field finetuning over all directed pairs, and an adversarial refinement that sharpens the output.

## 2 Materials and Methods

## 2.1 Dataset

We use the MRIxFields2026 multi-field brain-MRI dataset, which spans five field strengths (0.1T, 1.5T, 3T, 5T, 7T) acquired on scanners from three manufacturers, with three contrasts per field: T1w, T2w, and T2-FLAIR. The data comprise two cohorts. A retrospective, multi-centre cohort provides unpaired volumes, in which each subject is scanned at a single field strength, and where some subjects are missing contrasts. A prospective travelling cohort provides paired volumes, in which the same subject is scanned at every field strength, yielding spatially registered source and target pairs. The travelling training set for this challenge comprises only three subjects, each scanned with all three contrasts at all five field strengths – the central data constraint of our setting. Task 3 must be solved by a single set of parameters: with five fields there are $5 \times 4 = 2 0$ directed pairs, and with three contrasts $2 0 \times 3 = 6 0$ subtasks, and the rules forbid an ensemble or router of per-pair specialists. Submissions are scored on voxel metrics only, normalised RMSE (nRMSE), structural similarity (SSIM) [15], and LPIPS [16], over the central axial slab (slices [150, 180)) of each volume. The validation cohort ships inputs only, with cross-field ground truth withheld on the server, so predictions cannot be scored locally and are instead inspected qualitatively (Sect. 3).

## 2.2 Conditional flow matching

Conditional flow matching (CFM) [7] learns to transport an initial distribution π<sub>0</sub> to a target $\pi _ { 1 }$ along a smooth path, by regressing a time-dependent velocity field that carries samples $x _ { 0 } \sim \pi _ { 0 }$ to $x _ { 1 } \sim \pi _ { 1 }$ . Like difusion, it is trained by regression, but unlike the stochastic sampling of difusion [4] it defines a deterministic transport and generates samples by integrating an ODE. The network $v _ { \theta } ( x , t )$ is regressed onto the velocity of a conditional path; we take the linear interpolation between paired endpoints $x _ { 0 } \sim \pi _ { 0 }$ and $x _ { 1 } \sim \pi _ { 1 }$

$$
x _ { t } = \left( 1 - t \right) x _ { 0 } + t x _ { 1 } , \quad t \sim \mathcal { U } [ 0 , 1 ] .\tag{1}
$$

The velocity of this path is constant and given by its time derivative, ${ \dot { x } } _ { t } = x _ { 1 } - x _ { 0 }$ which serves as the regression target [14,8]. The network is trained by minimising

$$
\mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } } \left. v _ { \theta } ( x _ { t } , t ) - ( x _ { 1 } - x _ { 0 } ) \right. ^ { 2 } .\tag{2}
$$

Because the paired volumes are spatially registered, we bridge directly between the source slice $( x _ { 0 } )$ and the target slice $( x _ { 1 } )$ instead of noise. The source enters only at the $t = 0$ state of the integration, so the network sees the interpolated $x _ { t }$ and never receives the source as a separate input, keeping the bridge leak-free.

## 2.3 Neural network conditioning

The three contrasts are co-registered per subject and stacked as input channels, so the network predicts a three-channel velocity in a single pass. Because one set of parameters must serve every directed pair, the network cannot infer the intended translation from the input alone and is told which fields it maps between through three signals: the timestep t, the source field $s ,$ and the target field τ. The timestep uses a sinusoidal embedding and $s , \tau$ two learned embedding tables, summed into a single conditioning vector:

$$
c = \mathrm { e m b } _ { t } ( t ) + \mathrm { e m b } _ { s } ( s ) + \mathrm { e m b } _ { \tau } ( \tau ) .\tag{3}
$$

The conditioning vector c modulates the network through feature-wise linear modulation (FiLM) [11]. Inside each residual block, a single linear map projects c and splits its output into a per-channel scale $\gamma ( c )$ and shift $\beta ( c )$ , which are applied to the group-normalised feature map $h$

$$
\mathrm { F i L M } ( h \mid c ) = { \bigl ( } 1 + \gamma ( c ) { \bigr ) } \odot \mathrm { n o r m } ( h ) + \beta ( c ) ,\tag{4}
$$

where $\odot$ broadcasts the per-channel scale over spatial dimensions. The scale weights are zero-initialised, so modulation starts at identity for a stable start.

## 2.4 Inference

At inference we translate a source slice by integrating the learned velocity field forward in time, from the source at $t = 0$ to the target at $t = 1$ . The source slice is the initial condition ${ x _ { t = 0 } } = { x _ { 0 } }$ . We discretise [0, 1] into N uniform steps and integrate with Heun’s method, a second-order predictor–corrector scheme:

$$
\begin{array} { r } { \tilde { x } _ { t + h } = x _ { t } + h v _ { \theta } ( x _ { t } , t , s , \tau ) , } \end{array}\tag{5}
$$

$$
\begin{array} { r } { x _ { t + h } = x _ { t } + \frac { h } { 2 } \left[ v _ { \theta } ( x _ { t } , t , s , \tau ) + v _ { \theta } ( \tilde { x } _ { t + h } , t + h , s , \tau ) \right] , } \end{array}\tag{6}
$$

where the Euler step $\tilde { x } _ { t + h }$ is the predictor and the update averages the velocities at the current and predicted states. This tracks the trajectory more accurately than a single Euler step, so far fewer steps are needed and inference stays cheap; the same solver choice proved efective for difusion ODEs by Karras et al. [6].

## 2.5 Adversarial refinement

Regressing the velocity under a squared-error objective yields a mean estimator: wherever the mapping is uncertain, the network averages the plausible highfrequency detail, which appears as blur. We counter this with a generative adversarial [2] term, rewarding the generator (in our case: the flow matching model) for producing samples a discriminator D cannot tell from real, which restores realistic texture the CFM output lacks.

We use a conditional, multi-scale PatchGAN discriminator [5], which emits a grid of judgments over local receptive fields and so scores texture rather than global anatomy. It is conditioned on the same signals as the generator, the source $x _ { 0 }$ and the fields $s , \tau ,$ so it judges plausibility for that specific translation. It is trained with a hinge adversarial loss, and stabilised by a feature-matching [13] term on D’s intermediate activations. Applied only as a final stage on top of the trained CFM model, the refinement keeps a standard velocity-regression step alongside the adversarial and feature-matching terms; this faithfulness anchor ties the output to the correct translation and stops the adversarial pressure from drifting of target.

## 2.6 Staged training

We train in three stages: degradation-bridge pretraining on the unpaired retrospective cohort, then cross-field finetuning, and adversarial refinement on the paired travelling cohort [1]. The first stage builds a field conditioned prior from the abundant unpaired data; the latter two adapt that prior to the directed source-to-target mapping using the highly limited paired data.

The first stage, degradation-bridge pretraining (Algorithm 1), trains a restoration prior on the unpaired retrospective data. We bridge from a degraded copy of the target to the clean target: for a real slice x<sub>1</sub> at field τ we synthesise a corrupted $x _ { 0 } = \mathcal { D } ( x _ { 1 } )$ and regress the conditional flow matching velocity that transports $x _ { 0 } \ \mathrm { t o } \ x _ { 1 }$ . The source field is left unspecified (∅), and a per-channel mask m confines the loss to the contrasts a given subject actually has.

The operator D is an array of MRI motivated corruptions, each drawn and applied independently per sample with probability 0.4–0.5: additive noise $( \sigma \in [ 0 . 0 2 , 0 . 3 ] )$ , Gaussian blur $( \sigma \in [ 0 . 5 , 2 . 5 ] \mathrm { p x } )$ , resolution loss through downand up-sampling (factor 1.5–4), gamma ([0.4, 1.8]) and linear gain/bias shifts (gain [0.85, 1.2], bias $[ - 0 . 1 , 0 . 1 ] )$ , and a smooth multiplicative bias field (strength [0.1, 0.4]). Corruption is confined to the brain, leaving the skull-stripped background exactly at zero, and for 10% of samples $x _ { 0 }$ is replaced by pure noise so that the prior also learns to generate from scratch.

The second stage, cross-field finetuning (Algorithm 2), learns the directed source-to-target mapping from the paired travelling cohort, iterating over all ordered field pairs $s  \tau$ with $s \neq \tau$ . Here the bridge runs between two real acquisitions of the same subject, the source slice as $x _ { 0 }$ and the target as $x _ { 1 }$ . The source-field embedding $\operatorname { e m b } _ { s } ( s )$ is introduced and trained at this stage, while the remaining weights are warm-started from the pretrained prior.

The third stage, adversarial refinement (Algorithm 3), initialises the generator from the finetuned model. To produce a slice it integrates the learned velocity outward from the source with Euler steps to $t = 1$ , yielding ${ \hat { x } } _ { 1 } ;$ this rollout is never conditioned on the target and therefore matches the inference procedure closely. Since Heun’s method requires two network calls per step, in this training stage we use the simpler Euler method. The generator’s gradient is backpropagated through the whole integration trajectory, so the adversarial signal from the discriminator refines every step of the transport rather than the endpoint alone.

```tcl
Algorithm 1 Degradation-bridge pre- Algorithm 2 Cross-field finetuning
training (degraded target → target) (source → target)
Input: unpaired slices; degradation Input: paired slices, all directed pairs
menu $\mathcal { D } ;$ network $v _ { \theta } ; \mathrm { E M A } \ \bar { \theta }$ Initialise θ from pretraining $\mathrm { E M A } \ \bar { \theta }$
for each minibatch $( x _ { 1 } , \tau , m )$ do for each minibatch $( x _ { 0 } , x _ { 1 } , s , \tau )$ do
$x _ { 0 } \gets m \odot D ( x _ { 1 } )$ $t \sim \mathcal { U } [ 0 , 1 ]$
$t \sim \mathcal { U } [ 0 , 1 ]$ $x _ { t } \gets ( 1 - t ) x _ { 0 } + t x _ { 1 }$
$x _ { t } \gets \left( 1 - t \right) x _ { 0 } + t x _ { 1 }$ $v  v _ { \theta } ( x _ { t } , t , s , \tau )$
$v  v _ { \theta } ( x _ { t } , t , \emptyset , \tau )$ $\mathcal { L } _ { \mathrm { C F M } } \dot {  } \| v - ( x _ { 1 } - x _ { 0 } ) \| ^ { 2 }$
$\mathcal { L } _ { \mathrm { C F M } } \gets \lVert m \odot ( v - ( x _ { 1 } - x _ { 0 } ) ) \rVert ^ { 2 } / \sum m$ $\theta \gets \mathrm { A d a m W } ( \theta , \nabla _ { \theta } \mathcal { L } _ { \mathrm { C F M } } )$
$\underline { { \theta } } \gets \mathrm { A d a m } \mathrm { W } ( \theta , \nabla _ { \theta } \mathcal { L } _ { \mathrm { C F M } } )$ $\bar { \theta } \gets \mathrm { E M A } ( \bar { \theta } , \theta )$
$\bar { \theta } \gets \mathrm { E M A } ( \bar { \theta } , \theta )$ return <sup>¯</sup>θ
return <sup>¯</sup>θ
```

Algorithm 3 Adversarial refinement   
Input: paired slices; discriminator $D ;$ rollout steps $N _ { r } ;$ weights $\lambda _ { \mathrm { a d v } } , \lambda _ { \mathrm { f m } }$   
Initialise θ from finetuning EMA <sup>¯</sup>θ; initialise discriminator $D _ { \phi }$ with parameters $\phi$   
for each minibatch $( x _ { 0 } , x _ { 1 } , s , \tau )$ do   
$\hat { x } _ { 1 } \gets x _ { 0 }$ ▷ leak-free rollout: never sees $x _ { 1 }$   
for $i = 0$ to $N _ { r } - 1$ do   
$\begin{array} { r } { \hat { x } _ { 1 }  \hat { x } _ { 1 } + \frac { 1 } { N _ { r } } v _ { \theta } \big ( \hat { x } _ { 1 } , { \frac { i } { N _ { r } } } , s , \tau \big ) } \end{array}$ ▷ $N _ { r }$ Euler steps   
$\bar { \mathcal { L } } _ { D } \gets \mathrm { h i n g e } \big ( D _ { \phi } ( x _ { 1 } , x _ { 0 } , s , \tau ) , ~ D _ { \phi } ( \hat { x } _ { 1 } , x _ { 0 } , s , \tau ) \big )$   
$\phi \gets \mathrm { A d a m W } ( \phi , \nabla _ { \phi } \mathcal { L } _ { D } )$   
$t \sim \mathcal { U } [ 0 , 1 ] ; \quad x _ { t } \gets ( 1 - t ) x _ { 0 } + t x _ { 1 }$   
$\mathcal { L } _ { \mathrm { C F M } } \gets \left| \left| v _ { \theta } ( x _ { t } , t , s , \tau ) - ( x _ { 1 } - x _ { 0 } ) \right| \right| ^ { 2 }$ ▷ normal training step like Alg. 2   
$\mathcal { L } _ { \mathrm { a d v } }  -$ mean $D _ { \phi } \big ( \hat { x } _ { 1 } , x _ { 0 } , s , \tau \big )$   
$\mathcal { L } _ { \mathrm { f m } } \gets \left. D _ { \mathrm { f e a t } } ( \hat { x } _ { 1 } ) - D _ { \mathrm { f e a t } } ( x _ { 1 } ) \right. _ { 1 }$ ▷ feature matching   
${ \mathcal { L } } _ { G } \gets \dot { \lambda _ { \mathrm { C F M } } } { \mathcal { L } } _ { \mathrm { C F M } } + \lambda _ { \mathrm { a d v } } { \mathcal { L } } _ { \mathrm { a d v } } + \lambda _ { \mathrm { f m } } { \mathcal { L } } _ { \mathrm { f m } }$   
$\theta \gets \mathrm { A d a m W } ( \theta , \nabla _ { \theta } \mathcal { L } _ { G } ) ; \quad \bar { \theta } \gets \mathrm { E M A } ( \bar { \theta } , \theta )$   
return <sup>¯</sup>θ

## 3 Experiments and Results

## 3.1 Implementation details

Architecture: The velocity network is a 2D U-Net [12] (6.3M parameters) with four resolution levels ([32, 64, 128, 128] channels) and two residual blocks per level [3]. The adversarial discriminator is a two-scale conditional PatchGAN [5]; each scale is a three layer convolutional network (16 base channels) with a $3 4 \times 3 4$ receptive field.

Data handling: We work in 2D, treating 3D volumes as stacks of 2D axial slices. A slice is kept for training only if its intensity standard deviation exceeds a threshold, which discards the empty slices outside the brain. Each slice is min– max normalised to [0, 1] and rescaled to $[ - 1 , 1 ]$ for the network. All computations happen in [−1, 1]; outputs are mapped back to [0, 1] only when written out.

Optimisation: Every stage uses AdamW [9] with bf16 mixed precision and TF32 matmuls, and keeps an exponential moving average of the weights (decay 0.999). Pretraining runs on the unpaired cohort at a learning rate of $3 \times 1 0 ^ { - 4 }$ with batch size 8 for 200,000 steps. Cross-field finetuning then runs on the paired cohort for 100,000 steps, dropping the learning rate to $5 \times 1 0 ^ { - 5 }$ at the same batch size. Adversarial refinement continues for a further $5 0 { , } 0 0 0$ steps at batch size 8: the generator (flow U-Net) learns at $1 \times 1 0 ^ { - 5 }$ while the discriminator uses $3 \times 1 0 ^ { - 4 }$ 2 and the discriminator is trained alone for a 5,000-step warm-up before generator updates begin. The generator loss combines the velocity regression (weight 1), the adversarial term (weight 10<sup>−4</sup>, linearly ramped from zero over the first 2,000 generator steps), and the feature-matching term (weight 10). All experiments ran on a single NVIDIA RTX 6000 GPU (48 GB) in PyTorch [10].

Baselines: We compare against two baselines that share our U-Net, the FiLM field conditioning, the data splits, and the two-stage training schedule, so that only the training objective difers. The first is a residual supervised regression trained with an L1 loss through the same degradation-bridge pretraining and paired finetuning; it translates in a single forward pass. The second is an ϵ- prediction difusion model with a cosine noise schedule, pretrained as a denoising prior on the unpaired cohort and finetuned to denoise a noised source towards its target; at inference the source is noised to t = 0.5 and denoised back in 50 deterministic DDIM steps.

## 3.2 Results and ablation

Table 1 reports the translation quality on the challenge evaluation set, scored by the challenge platform and averaged over all 20 directed field pairs and the three contrasts; each directed pair is evaluated on three held-out test subjects, a diferent trio per pair, and the platform reports aggregate metrics only. Per-contrast scores stay within 0.008 SSIM and 0.003 LPIPS of the average, so we report the contrast average throughout. As references we include an identity floor that copies the source unmodified, and the two baselines described above. Figure 2 shows a representative T2w translation across the field transition. The identity floor shows how far the registered pairing alone carries: copying the source already gives 0.836 SSIM. Three comparisons stand out in Table 1. First, the baselines: the direct regression comes close on SSIM (0.906 vs. 0.909) but lags on perceptual quality (LPIPS 0.110 vs. 0.089) and reconstruction error (nRMSE 0.246 vs. 0.227), and the difusion baseline trails on every metric. Since both baselines share our architecture, conditioning, data, and staged training recipe, this gap isolates the contribution of conditional flow matching itself. Second, the stage progression: the pretrained restoration prior alone, which never saw a field pair, already reaches 0.837 SSIM; paired finetuning adds 0.049, and adversarial refinement a further 0.024, with matching gains in LPIPS and nRMSE.

Table 1. Translation quality on the challenge evaluation set, averaged over the three contrasts and all 20 directed field pairs. Top: identity floor and baselines sharing our architecture. Middle: our model after each training stage. Bottom: the final model integrated with 1, 5, and 10 Heun steps. The submitted configuration is in bold.
<table><tr><td>Method</td><td>Sampler</td><td>SSIM ↑LPIPS↓nRMSE↓</td><td></td></tr><tr><td>Identity (copy source) Regression (same U-Net) 1 forward pass</td><td>0.836 0.906</td><td>0.157 0.110</td><td>0.522 0.246</td></tr><tr><td>Diffusion (same U-Net) 50 DDIM steps</td><td>0.896</td><td>0.121</td><td>0.249</td></tr><tr><td>Ours: pretrained only 5 Heun steps</td><td>0.837</td><td>0.147</td><td>0.468</td></tr><tr><td>Ours: + finetuning 5 Heun steps</td><td>0.885</td><td>0.103</td><td>0.268</td></tr><tr><td>Ours: + adv. refinement 1 Heun step</td><td>0.817</td><td>0.171</td><td>0.382</td></tr><tr><td>Ours: + adv. refinement 5 Heun steps</td><td>0.909</td><td>0.089</td><td>0.227</td></tr><tr><td>Ours: + adv. refinement 10 Heun steps</td><td>0.909</td><td>0.089</td><td>0.229</td></tr></table>

Integrated in a single Heun step the refined model collapses to 0.817, because refinement trains the velocity field through a multi-step rollout and so calibrates it for multi-step integration; five steps recover 0.909 and ten bring no further gain.

## 4 Discussion and Conclusion

We presented a conditional flow matching approach to cross-field brain-MRI harmonisation for the MRIxFields2026 challenge, addressing the any-to-any Task 3 with a single 6.3M model that bridges a registered source slice to its target under a conditioned velocity field, spanning all 20 directed field pairs across three contrasts. The design rests on a three-stage recipe: a degradation bridge distills the unpaired retrospective cohort into a restoration prior; cross-field finetuning then learns the directed mapping; and adversarial refinement sharpens the output while a velocity-regression anchor preserves fidelity. Finetuned on only three paired travelling subjects, the model reaches a mean SSIM of 0.909, LPIPS of 0.089 and nRMSE of 0.227 over every directed pair and contrast, with a single set of parameters. It outperforms a direct regression baseline and a noised-source difusion baseline sharing its architecture, conditioning, data, and recipe, tying the gain to the flow matching objective rather than the recipe or network. Distilling a restoration prior from the abundant unpaired cohort frees the scarce paired data to learn the directed mapping rather than the underlying image prior from scratch, and five Heun steps match ten, keeping inference cheap. The main limitation is that the model operates slice by slice in 2D and so does not enforce through-plane coherence. The adversarial refinement could in principle hallucinate texture; the faithfulness anchor keeps the reconstruction error improving under refinement, but an anatomy-focused validation remains future work. The clearest routes forward are a 2.5D or 3D formulation to recover that coherence, a broader set of simulated degradations that better matches real low-field physics, and a larger paired cohort as more travelling-subject data become available.

Target  
![](images/91bafe0a9e5de483f563faa08f2c14894de8b2677b65e831780ca4a9f26bba62.jpg)  
Fig. 2. A series of example outputs from Task 3. The y-axis is the source field and the x-axis the target field. Scans with the red border are the ground truths for each target field, so every other scan is a prediction by our method.

## 5 Acknowledgments

This publication is part of the project AI4AI with file number P22.017 of the research program Perspectief which is (partly) financed by the Dutch Research Council (NWO) under the grant TTW Perspectief 2022-2023. This work was performed using the SHARK HPC cluster provided by the LUMC.

## References

1. Chang, H., Shang, Y., Wang, H., Liang, Y., Wang, H., Wang, F., Niu, C., Lian, C.: Controllable flow matching for 3d contrast-enhanced brain MRI synthesis from noncontrast scans. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. pp. 119–128. Springer Nature Switzerland. https://doi.org/10. 1007/978-3-032-05325-1\_12

2. Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks (2014), https:// arxiv.org/abs/1406.2661

3. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition (2015), https://arxiv.org/abs/1512.03385

4. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models (2020), https: //arxiv.org/abs/2006.11239

5. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks (2018), https://arxiv.org/abs/1611.07004

6. Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of difusionbased generative models (2022), https://arxiv.org/abs/2206.00364

7. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling (2022). https://doi.org/10.48550/ARXIV.2210.02747, https://arxiv.org/abs/2210.02747

8. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. In: The Eleventh International Conference on Learning Representations (ICLR) (2023), https://arxiv.org/abs/2209.03003

9. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization (2019), https: //arxiv.org/abs/1711.05101

10. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S.: Pytorch: An imperative style, high-performance deep learning library (2019), https://arxiv.org/abs/1912.01703

11. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer (2017), https://arxiv.org/abs/1709.07871

12. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation (2015), https://arxiv.org/abs/1505.04597

13. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.: Improved techniques for training gans. In: Proceedings of the 30th International Conference on Neural Information Processing Systems. p. 2234–2242. NIPS’16, Curran Associates Inc., Red Hook, NY, USA (2016)

14. Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y.: Improving and generalizing flow-based generative models with minibatch optimal transport (2023). https://doi.org/10.48550/ARXIV.2302.00482, https://arxiv.org/abs/2302.00482

15. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861

16. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 586–595 (2018). https://doi.org/10.1109/CVPR.2018.00068