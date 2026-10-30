# Blind Video Denoising for Vocal Tract rtMRI using a Self-Supervised Ensemble

Julian Andres Hernandez-Potes , Tomas Arias-Vergara ´ , Member, IEEE, Daiqi Liu , Andreas Maier , Member, IEEE, and Paula Andrea Perez-Toro , Member, IEEE

Abstract—Real-time magnetic resonance imaging of the vocal tract makes speech articulation directly observable, but the tradeoff between temporal resolution and signal quality introduces substantial noise and obscures fine articulatory detail. Since clean references cannot be acquired for a moving vocal tract, supervised denoising approaches are not applicable in this setting. We propose a self-supervised denoising framework that combines spatiotemporal super-resolution learning with pseudo-supervision derived from a variance-stabilized non-local means filter. A dualmodel fusion strategy balances noise suppression and preservation of high-frequency articulatory structure. Experiments on the USC-75 multi-speaker dataset under natural noise conditions show that the ensemble raises the signal-to-noise ratio by 11.9 dB on unseen subjects while attaining more high-frequency articulatory content in the frames. Among tuned implementations of anisotropic diffusion, block-matching filtering, non-local means and pre-smoothed non-local means, every configuration that achieves comparable noise suppression does so at the cost of highfrequency content or of structural fidelity. Under severe added Gaussian, Rayleigh and Rician noise, peak signal-to-noise ratio improves by 7.6 to 11.8 dB and structural similarity roughly triples. An ablation shows that the variance-stabilized pseudosupervision is an essential element for denoising, and that the effect is attributable to the hard thresholding contained in the stabilization transform.

Index Terms—Denoising, model fusion, real-time MRI, selfsupervised learning, vocal tract imaging.

## I. INTRODUCTION

M <sup>AGNETIC</sup> <sup>resonance</sup> <sup>imaging</sup> <sup>(MRI)</sup> <sup>data</sup> <sup>are</sup> <sup>inher-</sup>ently susceptible to noise, particularly in real-time, fast, ently susceptible to noise, particularly in real-time, fast or experimental sequences, which degrades image quality [1]. Whereas denoising methods have been widely developed for brain, fetal and cardiac imaging, analogous efforts for the vocal tract remain relatively underexplored [2], [3]. Such limitation is particularly relevant for real-time MRI (rtMRI) of the vocal tract, where the trade-off between temporal resolution and signal-to-noise ratio (SNR) produces noisy magnitude frames that obscure fine articulatory structures, such as the tongue and velum, which are essential for accurate speech analysis.

With the rise of deep learning, MRI denoising has shifted towards learning-based schemes. As reported in the reviews by Chen et al. [2] and more recently by Nazir et al. [4], convolutional encoder-decoder architectures such as the U-Net have become the standard and account for the majority of the published work along with transformer-based models, offering superior capability in capturing global dependencies and preserving fine structural details amidst complex noise patterns, such as Rician noise. Nevertheless, a recurring limitation identified in the current literature is the heavy reliance on supervised training with large paired datasets. While effective for static anatomy [5], this dependence on clean targets presents a fundamental bottleneck for dynamic speech imaging, where physiological motion renders the acquisition of noise-free reference data physically and practically unfeasible.

Recent research instead exploits the internal correlations of high-dimensional acquisitions to build supervisory signals without ground truth. N2SR-M by Tian et al. [6] trains on spatially sub-sampled pairs from noisy data to restore fine structures via Super-Resolution (SR) constraints. Although it was formulated for multi-contrast MRI, the principle transfers to the temporal dimension of rtMRI, and it has been validated in quantitative MRI sequences [7] and in high-speed fluorescence imaging [8]. Complementarily, we believe that conventional analytical approaches remain indispensable for their explicit handling of noise statistics and spatial signal characteristics [9], [10]. Works such as that of Yang et al. [11] have addressed Rician noise reduction via a Pre-Smoothed Non-Local-Means (PSNLM) filter with variance stabilization, demonstrating that explicitly modeling noise characteristics is crucial for preserving anatomical detail in low-SNR regimes.

Motivated by these complementary strengths, we propose a novel self-supervised ensemble for vocal tract rtMRI denoising that combines the spatiotemporal SR strategy of N2SR-M with the noise stabilization of PSNLM. Unlike single-frame blindspot networks [12]–[14] that rely solely on spatial context, our method exploits the spatiotemporal redundancy of speech sequences to disentangle stochastic noise from deterministic articulatory motion, achieving robust denoising without the need for a clean ground truth.

## II. DATASET

We use the USC-75 dataset [15], a multi-speaker public collection of raw and reconstructed 2-D sagittal rtMRI videos capturing speech production at the vocal tract in 75 healthy speakers. The reconstructed magnitude videos have a spatial resolution of 84 × 84 pixels and a temporal resolution of 83 frames/s. Consecutive frames are strongly correlated [16], [17], so we decimate the temporal axis by a factor of 4, which reduces the computational cost while preserving articulatory events at the time scale of interest. Rician noise and reconstruction artifacts are present in every available magnitude frame, and no noise-free counterpart exists.

![](images/546fa7f04c9d96c4c1845096b75f6f502d6284a0af337e4aabd690519adb923a.jpg)  
Fig. 1. Training framework for our denoising network. The noisy rtMRI video is cut into overlapping clips; the spatial sub-sampling block retains one pixel from every $2 \times \hat { 2 }$ cell, which forms the network input, and marks the remaining three quarters in the mask M. The network, detailed in Sec. III-C, is optimized against a PSNLM-filtered version $y ^ { p }$ of the same clip with a composite loss that preserves edges and limits over-blurring. Only the fidelity term is restricted to M. Eq. (5) defines all three terms of the loss function.

The proposed networks were trained on a pool of 54 subjects, and all results reported below are measured on subjects outside that pool. Four clips from four unseen subjects reading the Grandfather passage are reserved for selecting the hyperparameters of the analytical filters, so that no baseline is tuned on the data used for its evaluation. The remaining 17 unseen subjects, all reading the North Wind passage, form the test set.

## III. METHODS

We formulate rtMRI denoising as a self-supervised problem where no clean ground truth exists. Let $\dot { \boldsymbol { X } } ~ \in ~ \mathbb { R } ^ { \dot { \bar { T } } \times H \times W }$ denote a magnitude video corrupted by signal-dependent noise, segmented into the overlapping clips $x ^ { s e g }$ of Fig. 1 before being fed to the network.

## A. Spatiotemporal Self-Supervision

As illustrated in Fig. 1, we adopt a similar sub-sampling strategy as N2SR-M [6] to enable training without noise-free targets. The operator $\mathcal { \bar { D } } _ { s } : \mathbb { R } ^ { T \times H \times W }  \mathbb { R } ^ { T \times \frac { H } { 2 } \times \frac { W } { 2 } }$ partitions every frame into non-overlapping $2 \times 2$ cells and keeps one pixel per cell, drawn uniformly at random and independently for each cell and each frame. The retained pixels form the network input, while the remaining three quarters, collected in the mask M, are the only pixels on which the fidelity term of the loss is evaluated. Since the withheld pixel is not part of the input, its noise realization cannot be predicted from the observations, and the network is constrained to interpolate the underlying anatomy rather than to reproduce the noise of the target. The network $f _ { \theta }$ is thus trained to recover the fullresolution structure under the overall objective:

$$
\hat { \theta } = \arg \operatorname* { m i n } _ { \theta } ~ \mathbb { E } _ { X } \left[ \mathcal { L } \big ( f _ { \theta } \big ( \mathcal { D } _ { s } ( X ) \big ) , y ^ { p } \big ) \right] ,\tag{1}
$$

where $y ^ { p }$ is the pseudo-target defined in the following subsection. Because the sampled positions differ from frame to frame, reconstructing the withheld pixel requires not only aggregating information from spatial neighbors but also from adjacent frames.

## B. Variance-Stabilized Pseudo-Supervision

In a magnitude image the noise follows a Rician distribution. For an underlying signal amplitude A and Gaussian noise of standard deviation σ in each of the two orthogonal components of the complex-valued MRI signal:

$$
M _ { \mathrm { o b s } } = { \sqrt { ( A + n _ { 1 } ) ^ { 2 } + n _ { 2 } ^ { 2 } } } , \quad n _ { 1 } , n _ { 2 } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } ) ,\tag{2}
$$

so that $\mathbb { E } [ M _ { \mathrm { o b s } } ^ { 2 } ] = A ^ { 2 } + 2 \sigma ^ { 2 }$ and both the mean and the variance depend on the underlying SNR $r = A / \sigma$ . In air and inside the airway, where $A \ \approx \ 0 .$ , the magnitude follows a Rayleigh distribution with mean $\sigma { \sqrt { \pi / 2 } }$ . The noise in these regions does not have zero mean, so a raw noisy frame is a biased regression target in the regions that contain the articulatory structures of interest. Each frame is therefore filtered in a domain in which the noise is approximately additive and homoscedastic, and the result is used as the regression target:

$$
y _ { t } ^ { p } = F ^ { - 1 } ( \operatorname { P S N L M } ( F ( X _ { t } ) ) ) , \quad t = 1 , \dots , T ,\tag{3}
$$

with $F$ a variance-stabilizing transform (VST), $F ^ { - 1 }$ its inverse and PSNLM a pre-smoothed non-local means filter.

The choice of $F$ is not established for magnitude MRI, so two transforms are considered and compared in the ablation of Sec. IV-D. The first, $F _ { \mathrm { V } }$ , is the transform of Foi [18] obtained by quadrature over the Rice density so that the transformed data have the same standard deviation at every SNR, and inverted with the exact unbiased inverse. In our implementation the output standard deviation deviates by at most 0.8 % from a constant over $0 ~ \leq ~ r ~ \leq ~ 1 2$ , compared with 34 % for the untransformed magnitude. Its noise level is estimated per frame from the magnitudes $M _ { \mathrm { b g } }$ inside a background rectangle, where $A \approx 0$ and the Rayleigh median gives $\hat { \sigma } ~ = ~ \mathrm { m e d i a n } ( M _ { \mathrm { b g } } ) / \sqrt { 2 \ln { 2 } }$ . The second, $F _ { \mathrm { T } } .$ , is a closed-form algebraic approximation and is the transform used to generate the pseudo-targets of the models reported here. It is applied pixel by pixel and combines a Gudbjartsson–Patz [1] bias correction with an Anscombe-type root:

$$
F _ { \mathrm { T } } ( M _ { \mathrm { o b s } } ) = 2 \sqrt { \operatorname* { m a x } ( M _ { \mathrm { o b s } } ^ { 2 } - \hat { \sigma } _ { \mathrm { O } } ^ { 2 } , 0 ) + \frac { 3 } { 8 } } ,\tag{4}
$$

here $\scriptstyle { \hat { \sigma } } _ { \mathrm { O } }$ is the standard deviation of the pixels below an Otsu threshold, and the transform is inverted by $F _ { \mathrm { T } } ^ { - 1 } ( u ) =$ $\sqrt { \operatorname* { m a x } ( ( u / 2 ) ^ { 2 } - \textstyle \frac { 3 } { 8 } + \hat { \sigma } _ { \mathrm { O } } ^ { 2 } , 0 ) }$

## C. Network Architecture

We design a spatiotemporal U-Net with a pre-activation scheme [19], consisting of four main components:

1) Spatiotemporal Feature Extraction: the backbone consists of 3-D residual blocks enhanced with squeezeand-excitation (SE) modules [20] to model channel-wise dependencies.

2) Anti-Aliasing Downsampling: strided convolutions are replaced with non-learnable Gaussian blur filters to mitigate aliasing artifacts.

3) SR Upsampling: as in N2SR-M, reconstruction is performed by a SR block of 3-D transposed convolutions followed by projection-based refinement layers.

4) Temporal-Only Attention: a bottleneck attention block that focuses strictly on temporal dependencies to enforce frame-to-frame consistency.

The encoder has four stages with channel multipliers (1, 2, 4, 8) and two residual blocks per stage, and the SR block outputs twice the input grid in each spatial dimension.

## D. Composite Loss and Dual-Model Fusion

The network is optimized with a composite loss enforcing data fidelity, edge preservation, and smoothness. With xˆ the network output and M the mask of pixels withheld by $\mathcal { D } _ { s }$ :

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c h a r } } ( M [ \hat { x } ] , M [ y ^ { p } ] ) + \lambda _ { 1 } \mathcal { L } _ { G } ( \hat { x } , y ^ { p } ) + \lambda _ { 2 } \mathcal { L } _ { \mathrm { T V } } ( \hat { x } ) , } \end{array}\tag{5}
$$

where $\begin{array} { r } { \mathcal { L } _ { \mathrm { c h a r } } = \frac { 1 } { | M | } \sum _ { i \in M } { \sqrt { ( M [ \hat { x } ] _ { i } - M [ y ^ { p } ] _ { i } ) ^ { 2 } + \epsilon ^ { 2 } } } } \end{array}$ denotes the Charbonnier penalty over masked pixels, $\mathcal { L } _ { G } = \Vert \nabla \hat { x } -$ $\nabla y ^ { p } \| .$ <sub>1</sub> compares Sobel gradient magnitudes to promote sharp boundaries, and ${ \mathcal { L } } _ { \mathrm { T V } }$ is the total variation (TV) loss of the output to control smoothness.

We observed that a single value of $\lambda _ { 1 }$ imposes a compromise between texture preservation and noise suppression, so we train two networks with different values and fuse their outputs. $f _ { \mathrm { H F } }$ uses 16 base channels, 3-frame clips and $\lambda _ { 1 } ~ = ~ 1 . 2$ , which favors high-frequency detail and edges; $f _ { \mathrm { L F } }$ uses 48 base channels, 40-frame clips and $\lambda _ { 1 } ~ = ~ 0 . 7 ,$ which favors global smoothness and the preservation of the original anatomical structure. The denoised video is their convex combination:

$$
X _ { d } = w f _ { \mathrm { H F } } ( x ^ { s e g } ) + ( 1 - w ) f _ { \mathrm { L F } } ( x ^ { s e g } ) ,\tag{6}
$$

which is both locally sharp and globally consistent. The weight w is not fitted; it is set manually to trade off noise suppression against detail retention. We use $w = 0 . 8$ on the original data, where the noise level is moderate, and $w = 0 . 1$ under severe synthetic corruption.

## E. Training and Inference

Both networks are trained with AdamW, a learning rate of $2 \cdot 1 0 ^ { - 4 }$ for $f _ { \mathrm { H F } }$ and $5 \cdot 1 0 ^ { - 4 } $ for $f _ { \mathrm { L F } }$ , and a plateau schedule on the validation loss. At inference the sub-sampling is removed and the full-resolution video is passed through both networks; overlapping clips are merged by averaging and the super-resolved output is resampled back to the acquisition grid before the fusion of Eq. (6). No filter is applied at inference; all values reported for the proposed models are raw network outputs.

## IV. RESULTS

We assess performance in two scenarios: the original USC-75 rtMRI data, which contain only the noise present at acquisition, and a controlled setting in which synthetic noise is added to the same videos. In both, the ensemble is benchmarked against representative analytical denoisers, after which the pseudo-supervision that drives the training objective is ablated.

## A. Evaluation protocol

On the original data no clean reference exists, so we use three reference-free measures. The first is a region-of-interest SNR:

$$
\mathrm { r S N R } = 1 0 \log _ { 1 0 } \frac { \mu _ { S } ^ { 2 } } { \left( \hat { \sigma } _ { \mathcal { N } } / \sqrt { 2 - \pi / 2 } \right) ^ { 2 } } ,\tag{7}
$$

38.9 dB   
12.02   
0.88

Input

![](images/ebded64cd82430b5fea9ffcb5c8ceac8dbcf19456137668f1a8d2c7a5d338429.jpg)  
rSNR 22.4 dB TV 9.36 SSIM 1.00

ADF  
![](images/d6203c50a7a483a7978e90903172eeeac80044a028cdbe7ad25a08cab1778a32.jpg)

NLM-3D  
![](images/aba21990465a946445a022cf250d53949aafb307603bc924b25d04f76eba7c6f.jpg)

BM4D-3D  
![](images/1c90308a7af6fde8aacbc16a521e85a1670434a1c439b7e1cfdf839ea5ef6cd8.jpg)  
27.9 dB 6.73 0.90

PSNLM<sub>T</sub>  
![](images/906125f897ac40804cc75bc1de19e3be5913493e523b4b03c617216bdf7b0496.jpg)

PSNLM<sub>V</sub>  
![](images/89ad6089b83f6f13faf229a6d320cd740df836d91183012fd71639f1ddf6bd0b.jpg)

![](images/25666539d54b2d1c1fb3374abcce50de3a4ebd09812eed1a17178b2d36891238.jpg)  
Fig. 2. One frame of the unseen sub075 clip. The top row is the full frame, with the signal region that forms the numerator of Eq. (7) (red) and the background region-of-interest that forms its denominator (blue). The bottom row displays a magnified view of the full frame (yellow); all printed metrics are 50-frame averages over the clip, not only the displayed frame. ADF quantizes the soft tissue into piecewise-constant regions, BM4D-3D removes the noise together with the textures, and PSNLM preserves the anatomy but flattens the background, while PSNLM changes the signal much less. The proposed ensemble is the only denoised output whose TV exceeds that of the input while improving SNR and preserving structural fidelity.

where $\mu _ { S }$ is the mean intensity over a fixed 30×30 vocal-tract window and $\hat { \sigma } _ { \mathcal { N } }$ the standard deviation over a fixed $2 8 \times 2 8$ background window in the lower-left corner of the frame, both drawn in Fig. 2. The second measure is TV, which is included because rSNR alone favors over-smoothed outputs. The third is Structural Similarity Index (SSIM) computed against the noisy input to quantify agreement with the input anatomy. For every method, the input included, 50 consecutive frames are scored per clip.

We benchmark against Perona–Malik anisotropic diffusion (ADF) [21], [22], the volumetric transform-domain filter BM4D [23], non-local means (NLM) [9], and the PSNLM filter [11] that supplies our pseudo-targets, the latter in both stabilizations of Sec. III-B: $\mathrm { P S N L M _ { T } }$ uses the thresholded $F _ { \mathrm { T } }$ and $\mathrm { P S N L M _ { V } }$ the Rician $F _ { \mathrm { V } }$ . BM4D, NLM and PSNLM are additionally run in a spatiotemporal form, marked 3-D, which filters the whole $T \times H \times W$ volume. Their hyperparameter grids were scored inside this same protocol on the tuning clips of Sec. II. For each filter we retain the configuration that maximizes rSNR subject to $\mathrm { S S I M } \geq 0 . 9 0 .$ Where no configuration satisfies this constraint, as is the case for $\mathrm { P S N L M _ { T } } .$ , the unconstrained rSNR optimum is reported and marked as such.

## B. Reference-Free Evaluation

Table I reports quality averaged over the 17 test subjects, and Fig. 3 the paired difference between each method and the input of that same subject. Fig. 2 illustrates one of these clips, with its own measurements printed underneath.

The two members and their fusion behave as designed. $f _ { \mathrm { L F } }$ yields the highest rSNR of the three by prioritizing global smoothness, at the cost of TV and of an SSIM of 0.56 with respect to the input, which reflects structural deviation caused by over-smoothing. f<sub>HF</sub> recovers sharper textures, with the largest TV gain among all methods considered, at an SSIM

TABLE I

REFERENCE-FREE QUALITY AVERAGED OVER THE 17 TEST SUBJECTS. [PSEUDO-TARGET] IS THE SETTING THAT GENERATES $y ^ { p }$ , [MAX rSNR] THE UNCONSTRAINED OPTIMUM OF THE SEARCH AND THE UNMARKED

ROWS ITS SSIM ≥ 0.90 OPTIMUM. f<sub>HF</sub> AND f<sub>LF</sub>: HIGH- AND LOW-FREQUENCY COMPONENTS OF THE NETWORK RESPECTIVELY.
<table><tr><td>Method</td><td>rSNR ↑</td><td>TV</td><td>SSIM ↑</td></tr><tr><td>Original (noisy)</td><td>21.5±2.4</td><td>9.32±0.99</td><td>1.00±0.00</td></tr><tr><td>ADF</td><td>28.2±4.6</td><td>8.98±0.94</td><td>0.89±0.02</td></tr><tr><td>BM4D (3-D)</td><td>26.7±4.1</td><td>7.05±0.64</td><td>0.92±0.02</td></tr><tr><td>NLM (3-D)</td><td>31.6±5.9</td><td>8.31±0.71</td><td>0.95±0.04</td></tr><tr><td>PSNLMT [pseudo-target]</td><td>35.7±9.7</td><td>9.96±1.12</td><td>0.57±0.04</td></tr><tr><td>PSNLMT [max rSNR]</td><td>40.0±9.9</td><td>5.81±0.70</td><td>0.64±0.04</td></tr><tr><td>PSNLMy [pseudo-target]</td><td>27.9±5.1</td><td>8.51±0.81</td><td>0.80±0.09</td></tr><tr><td>PSNLMy [tuned]</td><td>20.5±2.4</td><td>9.46±1.04</td><td>0.92±0.03</td></tr><tr><td>fHF (ours)</td><td>32.0±7.6</td><td>12.73±1.33</td><td>0.87±0.04</td></tr><tr><td>fLF (ours)</td><td>39.9±10.1</td><td>9.31±0.89</td><td>0.56±0.03</td></tr><tr><td>Ensemble (ours)</td><td>33.4±8.0</td><td>12.05±1.24</td><td>0.85±0.04</td></tr></table>

of 0.87. The ensemble balances the two effects, suppressing noise by 11.9 dB while increasing TV by 2.73 at an SSIM of 0.85, with visible improvements in fine articulatory structures such as the velum (Fig. 2). Across the test subjects, the rSNR gain decreases and the TV gain increases monotonically with w, from +18.2 dB and +0.32 at w = 0.1 to +11.2 dB and $+ 3 . 0 8 { \mathrm { ~ a t ~ } } w = 0 . 9 \quad$

With respect to the analytical denoisers ADF, NLM and BM4D, the latter two also in their 3-D forms, improve rSNR by between 5.2 and 11.4 dB, and reduce TV by 0.34 to 3.20 (Fig. 3). $\mathrm { P S N L M _ { T } }$ achieves the largest rSNR gain among the analytical filters, 14.3 dB at the setting used to generate the pseudo-targets and 20.1 dB in its 3-D form. At that setting it is also the only analytical operating point in Fig. 3 that increases TV, by 0.65. Both results are obtained at an SSIM of 0.57, that is, with substantial deviation from the input. The Rician stabilized $\mathrm { P S N L M _ { V } }$ shows the opposite behavior: it maintains an SSIM of 0.80 but suppresses only 6.4 dB and still reduces TV. No analytical filter achieves the three properties simultaneously; only the ensemble and its $f _ { \mathrm { H F } }$ member suppress noise, increase high-frequency content and remain close to the input. The additional gradient energy introduced by the ensemble remains correlated with the anatomy of the input with a 0.96 Pearson correlation, which indicates that it corresponds to recovered detail and not to added noise. The hyper-parameter search for the filters leads to the same conclusion: of the 5950 configurations scored on the tuning clips, 1870 match the structural fidelity of the ensemble, and none of these exceeds it in both rSNR and TV. The highest TV among them is 9.88, against 12.53 for the ensemble on the same clips, and no configuration in the entire search exceeds 10.27, for an input value of 9.59.

![](images/51ac3a89cdcefa8835822bb370980f0fe8aebeed7700dcdffadec4beed984915.jpg)  
Fig. 3. Noise suppression against detail retention over the 17 test subjects. Each point is a method minus the noisy input of the same subject, and the bars are the standard deviation of that paired difference across subjects. All PSNLM points are the analytical filter alone, without the network.

## C. Reference-Based Evaluation

We additionally evaluate the ensemble on USC-75 frames corrupted with Gaussian, Rayleigh, and Rician noise at $\sigma =$ 0.2, using three reference-based metrics against the uncorrupted frame: the peak signal-to-noise ratio (PSNR), the normalized root mean squared error (NRMSE) and SSIM, which here compares the output with that reference rather than with the noisy input. As shown in Fig. 4, the method suppresses severe contamination under all three distributions, with $w = 0 . 1$ to maintain the high SNR of $f _ { \mathrm { L F } }$ while injecting sufficient high-frequency content to sharpen articulatory edges. PSNR increases from 15.5 to 23.6 dB under Gaussian noise, from 11.4 to 23.2 dB under Rayleigh noise and from 13.3 to 20.9 dB under Rician noise, and SSIM increases from 0.19, 0.24 and 0.16 to 0.58, 0.63 and 0.52 respectively, an approximately threefold improvement. This indicates that the geometry of the articulators is largely preserved under severe degradation.

![](images/7a8f2fd02d390528e15a11f701491c337ed8232d89cf0cf21cdb1f0cf916ec30.jpg)  
Fig. 4. The ensemble at $w = 0 . 1$ on the original videos after Gaussian, Rayleigh and Rician noise is added at $\sigma = 0 . { \overset { \cdot } { 2 } }$ . Top: the corrupted input. Bottom: the output. The reference for the three metrics is the uncorrupted USC-75 frame on the left, which carries the scanner’s own Rician noise.

## D. Pseudo-Supervision Ablation

Both ensemble members were re-trained against three regression targets, the pseudo-target built with $F _ { \mathrm { T } }$ , the one built with $F _ { \mathrm { V } }$ and the raw noisy clip, holding architecture, optimizer, schedule, data subset, seeds and number of updates fixed, so that the regression target is the only difference. These runs are shorter than those behind Table I and their absolute values are correspondingly lower.

![](images/0cb55cfac81e9abf4f2e3ca562f8e453317c64034aaf81626cc8f790e4a50d58.jpg)  
Fig. 5. Ablation of the pseudo-supervision with both ensemble members retrained under a common setting except for the regression target. rSNR of f<sub>HF</sub>, $f _ { \mathrm { L F } }$ and their fusion, trained against the thresholded pseudo-target $y _ { \mathrm { T } } ^ { p } ,$ the Rician-VST pseudo-target $y _ { \mathrm { V } } ^ { p }$ , and the raw noisy clip, with the input’s own rSNR as the dashed line.

The results are shown in Fig. 5. For $F _ { \mathrm { T } }$ as pseudo-target the fused output reaches an rSNR of 37.1 dB, being reduced to 29.8 dB for $F _ { \mathrm { V } }$ , whereas training on the raw noisy clip gives 22.5 dB, indistinguishable from the 22.4 dB of its own input, at an SSIM of 0.98 with respect to that input. Without a filtered target the network does not denoise but reproduces its input. The sub-sampling of Sec. III-A is therefore not sufficient on its own, since it decouples only the Charbonnier term of Eq. (5) from the input noise. $\mathcal { L } _ { G }$ is evaluated on all pixels, so with a noisy target it drives the output to reproduce the gradient field of the noise, and neither term addresses the non-zero-mean Rician floor of Eq. (2). The validation metrics of the original full-scale trainings, which paired the same configuration with and without the pseudo-target, are consistent with this: relative to its own input, the pseudo-target arm gains 11.8 dB and the noisy-target arm loses 0.7 dB, at essentially unchanged TV (7.87 and 7.70 respectively). The pseudo-target arm also exceeds the pseudo-target on which it was trained by 4.1 dB, showing that it does not simply reproduce the filter.

TABLE II  
CONTRIBUTION OF EACH COMPONENT OF THE PSEUDO-TARGET TRANSFORM. EACH ROW RETRAINS $f _ { \mathrm { H F } }$ AGAINST THE THRESHOLD COMPONENT, THE ANSCOMBE ROOT COMPONENT, OR THEIR COMBINATION $F _ { \mathrm { T } }$ , COMPARED WITH THE RICIAN-STABILIZED TRANSFORM $F _ { \mathrm { { V } } }$ AND WITH NO PSEUDO-TARGET.
<table><tr><td>Pseudo-target</td><td>rSNR ↑</td><td> $\Delta$ </td><td>SSIM</td></tr><tr><td>Noisy input</td><td>22.4</td><td>n/a</td><td>1.00</td></tr><tr><td>None (raw noisy clip)</td><td>22.3</td><td>-0.1</td><td>0.97</td></tr><tr><td>Anscombe root only</td><td>22.7</td><td> $+ 0 . 3$ </td><td>0.88</td></tr><tr><td> $F _ { \mathrm { V } } ,$  Rician VST</td><td>28.9</td><td> $+ 6 . 5$ </td><td>0.92</td></tr><tr><td>Threshold only</td><td>33.0</td><td> $+ 1 0 . 6$ </td><td>0.88</td></tr><tr><td> $F _ { \mathrm { T } }$  (threshold + root)</td><td>35.6</td><td> $+ 1 3 . 2$ </td><td>0.86</td></tr></table>

Between the two pseudo-targets, $F _ { \mathrm { T } }$ yields an rSNR higher by 7.3 dB. Since $F _ { \mathrm { T } }$ flattens the region that forms the denominator of rSNR, this metric alone is not conclusive. On the frame-varying component of the background, which a fixed pattern cannot reduce, the $F _ { \mathrm { T } }$ arm lowers the standard deviation by a factor of 20.2, compared with 12.8 for $F _ { \mathrm { V } }$ and 4.0 for the noisy target, while a control that only thresholds the input, without filtering, gives 0.9. The $F _ { \mathrm { T } }$ arm also agrees better with the input structures inside the vocal tract giving a Pearson correlation of 0.95 against 0.87.

Table II shows which of the two operations composing $F _ { \mathrm { T } }$ the hard threshold or the Anscombe root, accounts for its advantage over $F _ { \mathrm { V } }$ . This is a separate, smaller experiment than the one above: only the $f _ { \mathrm { H F } }$ member is retrained, against a pseudo-target built from each half of Eq. (4) in isolation, with the non-local means stage, the noise estimator, the data, the schedule and the seed held fixed as before. The threshold alone reaches 33.0 dB, most of the 35.6 dB obtained with the full transform, whereas the Anscombe root alone reaches 22.7 dB, the same level as the arm trained without a pseudo-target at all. The supervision therefore derives from the hard thresholding rather than from the variance stabilization alone for which the transform was initially introduced.

## V. CONCLUSION

We presented a self-supervised ensemble for blind vocal tract rtMRI denoising that requires no clean ground truth. By combining spatiotemporal SR training with a stabilized pseudo-supervision and a dual-model fusion strategy, the method effectively separates signal-dependent noise from dynamic articulatory structures across the USC-75 dataset. The ensemble mitigates the smoothness-sharpness trade-off inherent in single-model approaches. On the original data, a balanced fusion enhances high-frequency articulatory detail without distorting global anatomy, while under severe synthetic degradation, a stronger low-frequency contribution drives robust structural recovery. In both cases, the ensemble yields quantitatively and qualitatively superior results among the compared filters, demonstrating the viability of purely selfsupervised methods for vocal tract rtMRI applications where clean reference data is physically unattainable.

A comparison against different classical filters, searched exhaustively over their hyper-parameters, shows that suppressing the noise floor while increasing high-frequency articulatory content at comparable structural fidelity is not reachable by re-tuning the filters. The ablation shows that the pseudosupervision, rather than the super-resolution constraint alone, drives the denoising, and that its effect comes from the hard threshold in $F _ { \mathrm { T } }$ rather than from the variance-stabilizing root.

## ACKNOWLEDGMENT

We gratefully acknowledge the scientific support and computational resources provided by the Erlangen National High-Performance Computer Center (NHR@FAU) of the Friedrich-Alexander-Universitat Erlangen-N¨ urnberg.¨

## REFERENCES

[1] Gudbjartsson H., Patz S., “The rician distribution of noisy MRI data,” Magn Reson Med., 1995.

[2] Chen, Z., Pawar, K., Ekanayake, M. et al., “Deep learning for image enhancement and correction in magnetic resonance imaging—state-ofthe-art and challenges,” Journal of Digital Imaging, 2022.

[3] V. Ramanarayanan, S. Tilsen, M. Proctor, J. Toger, L. Goldstein, K. S.¨ Nayak, and S. Narayanan, “Analysis of speech production real-time MRI,” Computer Speech & Language, vol. 52, 2018.

[4] N. Nazir, A. Sarwar, and B. S. Saini, “Recent developments in denoising medical images using deep learning: An overview of models, techniques, and challenges,” Micron, vol. 180, p. 103615, 2024.

[5] M. Ran, J. Hu, Y. Chen, H. Chen, H. Sun, J. Zhou, and Y. Zhang, “Denoising of 3-D Magnetic Resonance Images Using a Residual Encoder-Decoder Wasserstein Generative Adversarial Network,” 2019. [Online]. Available: https://arxiv.org/abs/1808.03941

[6] X. Tian, J. Wu, G. Lao, C. Du, C. Jiang, Y. Li, J. L. Zhang, H. Wei, and Y. Zhang, “Self-supervised denoising for high-dimensional magnetic resonance image,” Biomedical Signal Processing and Control, vol. 104, p. 107451, 2025.

[7] B. Kang, W. Lee, H. Seo, H.-Y. Heo, and H. Park, “Self-supervised learning for denoising of multidimensional MRI data,” Magnetic Resonance in Medicine, vol. 92, no. 5, pp. 1980–1994, 2024. [Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.30197

[8] Y. Wang, Y. Gu, J. Wang, A. Xuan, C. Kong, W.-Q. Fang, D. Li, D. Zhu, F. Ding, and B. Dong, “Real-time self-supervised denoising for highspeed fluorescence neural imaging,” Nat. Commun., vol. 16, no. 1, p. 9396, Oct. 2025.

[9] H. V. Bhujle and B. H. Vadavadagi, “NLM based magnetic resonance image denoising – A review,” Biomedical Signal Processing and Control, vol. 47, pp. 252–261, 2019.

[10] S. Li, F. Wang, and S. Gao, “New non-local mean methods for MRI denoising based on global self-similarity between values,” Computers in Biology and Medicine, vol. 174, p. 108450, 2024.

[11] Yang, J., Fan, J., Ai, D. et al., “Brain MR image denoising for Rician noise using pre-smooth non-local means filter,” BioMed Eng OnLine, vol. 14, 2015.

[12] A. Krull, T.-O. Buchholz, and F. Jug, “Noise2void - learning denoising from single noisy images,” 2019. [Online]. Available: https://arxiv.org/abs/1811.10980

[13] J. Xu, Y. Huang, M.-M. Cheng, L. Liu, F. Zhu, Z. Xu, and L. Shao, “Noisy-as-clean: Learning self-supervised denoising from corrupted image,” IEEE Transactions on Image Processing, vol. 29, p. 9316–9329, 2020. [Online]. Available: http://dx.doi.org/10.1109/TIP.2020.3026622

[14] X. Tian, Q. Wu, H. Wei, and Y. Zhang, “Noise2sr: Learning to denoise from super-resolved single noisy fluorescence image,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2022, L. Wang, Q. Dou, P. T. Fletcher, S. Speidel, and S. Li, Eds. Cham: Springer Nature Switzerland, 2022, pp. 334–343.

[15] Lim, Y., Toutios, A., Bliesener, Y. et al., “A multispeaker dataset of raw and reconstructed speech production real-time MRI video and 3d volumetric images,” Scientific Data, 2021.

[16] S. G. Lingala, B. P. Sutton, M. E. Miquel, and K. S. Nayak, “Recommendations for real-time speech MRI,” Journal of Magnetic Resonance Imaging, vol. 43, no. 1, 2015.

[17] Z. Wu and Y.-G. Jiang, Deep learning for video understanding, 2024th ed. Cham, Switzerland: Springer International Publishing, Aug. 2024.

[18] A. Foi, “Noise estimation and removal in MR imaging: The variancestabilization approach,” in Proc. IEEE Int. Symp. Biomedical Imaging (ISBI), 2011, pp. 1809–1814.

[19] K. He, X. Zhang, S. Ren, and J. Sun, “Identity mappings in deep residual networks,” 2016. [Online]. Available: https://arxiv.org/abs/1603.05027

[20] J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, “Squeeze-and-excitation

networks,” 2019. [Online]. Available: https://arxiv.org/abs/1709.01507

[21] P. Perona and J. Malik, “Scale-space and edge detection using anisotropic diffusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 12, no. 7, pp. 629–639, 1990.

[22] G. Gerig, O. Kubler, R. Kikinis, and F. A. Jolesz, “Nonlinear anisotropic¨ filtering of MRI data,” IEEE Transactions on Medical Imaging, vol. 11, no. 2, pp. 221–232, 1992.

[23] M. Maggioni, V. Katkovnik, K. Egiazarian, and A. Foi, “Nonlocal transform-domain filter for volumetric data denoising and reconstruction,” IEEE Transactions on Image Processing, vol. 22, no. 1, pp. 119– 133, 2013.