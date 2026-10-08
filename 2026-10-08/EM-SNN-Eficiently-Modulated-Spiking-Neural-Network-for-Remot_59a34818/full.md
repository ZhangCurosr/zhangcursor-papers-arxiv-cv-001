# EM-SNN: Eficiently Modulated Spiking Neural Network for Remote Sensing Image Dehazing

Jie Shao<sup>1</sup>, Jiaqi Ma<sup>2</sup>, Wenwen Min<sup>3</sup>, Beihang Song<sup>4</sup>, Ning Chen<sup>1</sup>, Youfa Liu<sup>5</sup>, Jun Wan<sup>1∗</sup>

<sup>1</sup>Zhongnan University of Economics and Law, Wuhan, China

<sup>2</sup>Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE

<sup>3</sup>Yunnan University, Kunming, China

<sup>4</sup>National Institute of Natural Hazards, Ministry of Emergency Management of China, Beijing, China <sup>5</sup>Wuhan University, Wuhan, China

## Abstract

Although spiking neural networks (SNNs) provide an energyeficient alternative to artificial neural networks (ANNs), their application to remote sensing image dehazing remains limited. A key challenge arises from the coupling between hazeinduced high-frequency attenuation and discrete spike thresholding. This interaction suppresses weak responses and fundamentally limits the recovery of edges, textures, and fine details in spiking dehazing models. To address this challenge, we propose the Eficiently Modulated Spiking Neural Network (EM-SNN), a dedicated spiking framework tailored to remote sensing image dehazing. EM-SNN integrates a statisticsdriven Threshold-Modulated Leaky Integrate-and-Fire (TM-LIF) neuron to adaptively compensate for haze-induced contrast compression, together with a Spike Sobel Modulation (SSM) module that enhances structural cues and reduces depth-wise attenuation during spiking feature propagation. By jointly modulating activation scales and structural representations, EM-SNN improves dehazing performance while preserving the inherent event-driven sparsity of SNNs. Experiments on HRSD, RICE, RRSHID, and SateHaze1K demonstrate that EM-SNN achieves competitive dehazing performance while consuming only one quarter of the energy of the strong ANN baseline SFRDP-Net.

## Introduction

Remote sensing images provide detailed ground scene information and are widely used in numerous applications, including agricultural production (Mladenova et al. 2019), environmental protection (Jiang et al. 2022), and disaster monitoring (Zhang et al. 2002). Haze and atmospheric scattering degrade image quality by reducing contrast and obscuring structural details. This degradation is highly frequency-dependent, with high-frequency components (e.g., edges and textures) being severely attenuated while low-frequency components are largely preserved, resulting in over-smoothed images with lost details. Although deep learning models can recover such structures, they often come with a trade-of of increased computational overhead and growing storage costs, reducing their eficacy in practical scenarios (Sun et al. 2025; Hu et al. 2025).

More recently, brain-inspired SNNs, as the third generation of neural networks, have emerged as a promising alternative for energy-eficient intelligence. Unlike conventional

(a)  
(b)  
(c)  
![](images/9f6618c0b88753913939ed359233e217758033c25a805325108069cca196a819.jpg)  
Figure 1: Visualization of haze-induced degradation and spiking responses. (a) Remote-sensing patch under clear (top) and hazy (bottom) conditions. (b) 3D intensity surface plots of the patches. (c) LIF membrane potentials over neuron index and simulation step. (d) PSNR versus energy consumption on RICE-2; bubble size indicates parameter.

ANNs that encode features as continuous values, SNNs communicate with discrete spikes and propagate information through event-driven neuronal dynamics, facilitating deployment on neuromorphic hardware with mW-scale power budgets (Yao et al. 2024). These properties make SNNs particularly attractive for remote sensing dehazing under constrained onboard computation and power budgets.

However, extending SNNs to remote sensing image dehazing remains challenging. Remote sensing scenes contain abundant small objects, thin boundaries, and fine-grained textures, whose feature responses are particularly vulnerable to haze-induced attenuation. Haze compresses local contrast and shifts these weak structural responses into a lowactivation regime, resulting in compounded information loss when they are subsequently encoded by discrete spikes. Under fixed firing thresholds, the resulting amplitude–threshold mismatch causes many weak edge and texture activations to remain below the firing boundary, making them underrepresented or even discarded in the spike domain. Moreover, repeated spike quantization suppresses subtle activation variations, while membrane leakage makes weak responses more dificult to accumulate across time. These efects progressively attenuate high-frequency structural information as network depth increases. As shown in Fig. 1(a–c), compared with clear inputs, hazy inputs produce lower membranepotential responses and a larger proportion of sub-threshold activations, particularly around weak edges and textures.

Motivated by this analysis, we propose EM-SNN, an energy-eficient spike-driven framework for remote sensing image dehazing, composed of a Threshold-Modulated Leaky Integrate-and-Fire (TM-LIF) neuron and a Spike Sobel Modulation (SSM) module. The TM-LIF neuron calibrates spike quantization via a statistics-driven, channel-wise adaptive threshold, improving robustness under haze-induced contrast compression and stabilizing feature scales across depth. The SSM module uses Sobel gradients as a lightweight structural prior to perform channel and spatial modulation, compensating for the depth-wise attenuation of weak structural details during spiking feature propagation. Together, TM-LIF and SSM provide coupled scale-and-structure modulation for high-quality remote sensing image dehazing. As shown in Fig. 1(d), the proposed EM-SNN significantly reduces energy consumption while maintaining competitive PSNR compared with existing approaches. Our contributions are as follows:

• A TM-LIF neuron is proposed to calibrate spike quantization via a statistics-driven, channel-wise adaptive threshold for stabilizing multispike representations and mitigating information loss caused by haze-induced contrast compression.

• A deployment-friendly and energy-aware SSM module is designed to leverage Sobel-derived structural priors to compensate for the depth-wise attenuation of weak structural details in deep spiking feature propagation.

• By collaborating TM-LIF neuron and SSM module within the proposed EM-SNN, activation scales and structural representations are more eficiently modulated, achieving competitive dehazing performance with significantly reduced energy consumption.

## Related Work

## Single Image Dehazing

Prior-based Methods. Early single-image dehazing methods estimate atmospheric light and scene transmission using hand-crafted priors, including the Dark Channel Prior (DCP) (He, Sun, and Tang 2010), Color Attenuation Prior (Zhu, Mai, and Shao 2015), Color-Lines (Fattal 2014), and Hazelines (Berman, Avidan et al. 2016). Remotesensing-oriented variants further introduce homomorphic filtering and sphere models (Li, Hu, and Ai 2018), low-rank and sparse priors (Bi et al. 2021), or superpixel-based heterogeneous concentration priors (Liang et al. 2024). However, their performance often degrades when the underlying assumptions are violated.

Deep Learning Methods. Deep neural networks have become the dominant paradigm for image dehazing. Early CNN models, such as DehazeNet (Cai et al. 2016) and AOD-Net (Li et al. 2017), learn transmission estimation or atmospheric-model inversion, while later methods improve feature representation through attention and multiscale interaction, such as FFA-Net (Qin et al. 2020) and GridDehazeNet (Liu et al. 2019). Transformer-based models, including DehazeFormer (Song et al. 2023) and Trinity (Chi, Yuan, and Wang 2023), further enhance long-range dependency modeling. Recent remote sensing methods, such as SFRDP-Net (Sun et al. 2025) and DS-RDMPD (Zhou et al. 2025), exploit spatial–frequency fusion or coarse-to-fine difusion restoration. Despite their strong performance, these models often require substantial computation, whereas lightweight alternatives may sacrifice representation capacity. In contrast, we investigate spiking neural networks for energyeficient remote sensing image dehazing without compromising restoration quality.

## Spiking Neural Networks

Spiking neural networks (SNNs) have attracted increasing attention owing to their event-driven computation and potential advantages in latency and energy eficiency over conventional artificial neural networks (Fang et al. 2025; Lei et al. 2025). They have been applied to various vision tasks, including object detection, deraining, and super-resolution (Luo et al. 2024; Song et al. 2024; Xiao et al. 2025). Since spike generation is non-diferentiable, directly trained SNNs commonly employ surrogate gradients for backpropagation through time. Advances in membrane-potential optimization and temporal normalization have improved the trainability of deep SNNs (Lee, Delbruck, and Pfeifer 2016; Zheng et al. 2021a), while residual learning and attention mechanisms further enhance their spatial representation capability (Shi, Hao, and Yu 2024; Lee et al. 2025).

Nevertheless, pixel-level dense restoration with SNNs remains underexplored. Existing studies mainly focus on super-resolution and deraining, while atmospheric-scattering degradation has received limited attention. Previous SNNbased dehazing methods largely rely on spiking-inspired approximations rather than fully spike-driven restoration pipelines (Zhang et al. 2024b; Sudevan et al. 2026). Moreover, remote sensing haze is spatially heterogeneous and severely attenuates high-frequency information. The interaction between haze-induced contrast compression and spike discretization can further suppress weak edges and textures, leading to progressive structural-detail loss across network depth.

## Method

## Overall Architecture

In this paper, we introduce EM-SNN for remote sensing imagery. As illustrated in Figure 2, EM-SNN adopts an encoder–decoder architecture. Shallow features extracted from the input image by a 3 × 3 convolution are fed to a multiscale encoder with Spiking Residual Blocks (SRBs) interleaved with downsampling (Song et al. 2024). The proposed Spike Sobel Modulation (SSM) module incorporates Sobelderived structural priors into each SRB to enhance edges and textures. Throughout the encoder, spiking feature transformations are carried out through convolutional layers followed by Threshold-Modulated leaky integrate-and-fire (TM-LIF) neurons. In the decoder, Up modules with skip connections progressively recover the spatial resolution. Following (Song et al. 2024), a Prediction Reconstruction Block (PRB) converts decoded spike features into continuous representations; the final output is obtained by a 3 × 3 convolution.

![](images/0fb1fe77d4c5ed95df8358985df78eddd211467168e8822cb52fcdf3acf5a9e5.jpg)  
Figure 2: Overall architecture of EM-SNN. EM-SNN adopts a multi-scale encoder–decoder built from SRB blocks with skip fusion via channel-wise concatenation (C), and uses PRB for reconstruction. TM-LIF is applied in all SCB units across both SRB blocks and the Down/Upsampling stages, while SSM is embedded across scales to provide structure-guided attention.

## Threshold-Modulated LIF (TM-LIF)

Remote sensing dehazing is a dense prediction task that requires stable feature magnitudes and consistent activation scales across network depth. However, fixed-threshold LIF neurons are sensitive to haze-induced contrast compression and layer-wise distribution shifts, which may push weak responses below the firing boundary and cause progressive information attenuation and scale inconsistency in deep SNNs (Zhao et al. 2025). Normalized Integer LIF (NI-LIF) alleviates gradient instability by normalizing multi-leve spike counts with a fixed quantization factor D, thereby constraining spiking outputs to a bounded range (Lei et al. 2025). Nevertheless, fixed and layer-wise shared quantization boundaries are ill-suited to channel-wise scale heterogeneity, leading to quantization mismatch and imbalanced firing that weaken information transmission and representation in deep SNNs (Kim et al. 2020; Guo et al. 2023a,b). Motivated by these limitations, we propose TM-LIF, which replaces the fixed firing threshold with a statistics-calibrated, channelwise threshold. By aligning firing boundaries with the characteristic membrane-potential scales of individual channels, TM-LIF reduces scale mismatch and supports high-fidelity image restoration.

Statistics-Calibrated Firing Threshold. TM-LIF maintains a channel-wise running estimate of membrane-potential dispersion during training. At training iteration $\bar { k , }$ the running statistic is updated as

$$
\begin{array} { r } { \bar { \sigma } ^ { ( k ) } = \mu \bar { \sigma } ^ { ( k - 1 ) } + \left( 1 - \mu \right) \mathrm { S t d } _ { ( b , h , w ) } \left( \mathrm { s g } \left( U ^ { ( k ) } \right) \right) , } \end{array}\tag{1}
$$

where $U ^ { ( k ) } \in \mathbb { R } ^ { B \times C \times H \times W }$ denotes the pre-spike membrane potential, and $\mathrm { S t d } _ { ( b , h , w ) } ( \cdot )$ computes the standard deviation over the batch and spatial dimensions, yielding a channel-wise statistic in $\mathbb { R } ^ { 1 \times \dot { C } \times 1 \times 1 }$ . The momentum coefficient $\mu \in ( 0 , 1 )$ controls the exponential moving average, while $\operatorname { s g } ( \cdot )$ stops gradients through the statistic.

The channel-wise firing threshold is then defined as

$$
{ v } _ { \mathrm { t h } } ^ { ( k ) } = \alpha \bar { \sigma } ^ { ( k ) } ,\tag{2}
$$

where α controls the efective quantization interval. During training, the running statistics progressively capture the characteristic activation scales of diferent channels. Consequently, TM-LIF learns channel-specific quantization boundaries that reduce scale mismatch across layers and channels. Neuron Dynamics. Let $\mathbf { I } _ { t }$ denote the synaptic input at timestep t and $\mathbf { H } _ { t - 1 } \in \mathbb { R } ^ { \mathbf { \tilde { B } } \times C \times H \times W }$ the previous membrane state. Given the threshold ${ \pmb v } _ { \mathrm { t h } } ^ { ( k ) }$ , TM-LIF updates the pre-spike membrane potential and produces a multi-level firing output as:

$$
\mathbf { U } _ { t } = \beta \mathbf { H } _ { t - 1 } + \mathbf { I } _ { t } ,\tag{3}
$$

$$
\mathbf { S } _ { t } ^ { \mathrm { i n t } } = \mathrm { C l i p } \left( \operatorname { R o u n d } \left( \frac { \mathbf { U } _ { t } } { { v } _ { \mathrm { t h } } ^ { ( k ) } } \right) , 0 , D \right) ,\tag{4}
$$

$$
\mathbf { S } _ { t } = \frac { 1 } { D } \mathbf { S } _ { t } ^ { \mathrm { i n t } } ,\tag{5}
$$

$$
\mathbf { H } _ { t } = \mathbf { U } _ { t } - \mathbf { S } _ { t } ^ { \mathrm { i n t } } \odot { \boldsymbol { v } } _ { \mathrm { t h } } ^ { ( k ) } .\tag{6}
$$

Where $\mathbf { U } _ { t }$ integrates the leaked previous state and current input. $\mathbf { S } _ { t } ^ { \mathrm { i n t } } \in \{ 0 , \ldots , D \}$ is the integer firing level obtained by quantizing $\mathbf { U } _ { t }$ in units of ${ \pmb v } _ { \mathrm { t h } } ^ { ( k ) }$ (as illustrated by the threshold steps in the TM-LIF panel of Fig. 2), while $\mathbf S _ { t } \in \{ 0 , \frac { 1 } { D } , \dots , \overset { \cdot } { 1 } \}$ is the normalized output. $\bar { \beta } \in ( 0 , 1 )$ is the leak factor, and Round(·) and $\mathrm { C l i p } ( \cdot , \bar { 0 } , D )$ are applied element-wise.

Inference. After training, the accumulated running statistics are frozen and used for all test samples. Specifically, the inference-time threshold is given by

$$
{ v _ { \mathrm { t h } } ^ { * } } = \alpha \bar { \sigma } ^ { * } ,\tag{7}
$$

where $\bar { \sigma } ^ { * }$ denotes the running membrane-dispersion estimate obtained at the end of training. Therefore, the thresholds are channel-specific and conditioned on the training distribution, but remain fixed across test samples. This design avoids batch-dependent statistics and online threshold estimation during inference, leading to deterministic spike quantization without additional adaptation overhead.

Scale Stabilization and Gain Control of TM-LIF. TM-LIF can be interpreted as a statistics-calibrated re-scaling operation applied before the surrogate nonlinearity. For a mapping $y = f ( x )$ , local linearization gives $\Delta y \approx J \Delta x .$ , where $J = \partial y / \partial x$ . The spectral norm $\| J \| _ { 2 }$ bounds the worstcase amplification of local forward perturbations and backpropagated gradients. For clarity, we omit the simulation-step and channel indices in the following analysis. Our objective is to (i) maintain comparable pre-nonlinearity scales across layers and channels and (ii) provide an explicit mechanism for regulating the local gain $\Vert \boldsymbol J \Vert _ { 2 }$

(1) Scale calibration via running statistics. Consider a single layer with $u \ = \ W x$ and a channel-wise threshold $v _ { \mathrm { t h } } ~ = ~ \alpha \bar { \sigma }$ , where σ¯ denotes the running estimate of membrane-potential dispersion. Let

$$
q = \frac { u } { v _ { \mathrm { t h } } } , \qquad y = \varphi ( q ) ,\tag{8}
$$

where $\varphi ( \cdot )$ denotes the normalized multi-level firing mapping used with the surrogate gradient. When σ¯ closely tracks the characteristic dispersion of u, we have

$$
\mathrm { S t d } ( q ) = \frac { \mathrm { S t d } ( u ) } { \alpha \bar { \sigma } } \approx \frac { 1 } { \alpha } .\tag{9}
$$

Therefore, the input to $\varphi ( \cdot )$ remains within a comparable scale range across layers and channels, reducing scale mismatch before spike quantization.

(2) Explicit gain control through a Jacobian bound. Because gradients are stopped through the running statistic, $v _ { \mathrm { t h } }$ can be treated as locally constant during back-propagation. By the chain rule,

$$
J = \frac { \partial y } { \partial x } \approx \frac { 1 } { v _ { \mathrm { t h } } } \mathrm { d i a g } \left( \varphi ^ { \prime } ( q ) \right) W ,\tag{10}
$$

which yields

$$
\| J \| _ { 2 } \leq \frac { \| W \| _ { 2 } } { \alpha \bar { \sigma } } B _ { \varphi } ,\tag{11}
$$

where $B _ { \varphi } = \operatorname* { s u p } _ { q } | \varphi ^ { \prime } ( q ) |$ depends on the surrogate firing function but is independent of α. This local bound shows that α and the estimated channel-wise dispersion jointly regulate the efective gain of the surrogate mapping.

(3) Training-time calibration and frozen inference. During training, the running statistic progressively estimates the characteristic channel-wise dispersion of the membrane potential. At inference, both the accumulated statistic and its corresponding threshold are frozen, providing deterministic channel-wise quantization boundaries calibrated to the training distribution.

## Spike Sobel Modulation (SSM)

Haze attenuates high-frequency structural details, so edges and fine textures enter the network as low-amplitude, lowcontrast cues (Fig. 3). In SNNs, such weak cues become small and fast-varying membrane fluctuations under multistep dynamics and are thus further suppressed by the implicit temporal low-pass filtering of LIF subthreshold integration. Specifically, the discrete-time subthreshold update in Eq. (3) is a first-order recursion. Applying the Z-transform yields

![](images/6132bef4250f545e108616782633b15485a358021005b1c56a8f9c5fb1100366.jpg)  
Figure 3: Frequency-domain visualizations under diferent neurons. The left shows clear and hazy remote sensing images with the region of interest highlighted, while the right compares ReLU, LIF, and TM-LIF in terms of spectra. We report $\Delta ,$ the high-frequency energy ratio drop from clear to hazy, which indicates that haze suppresses high-frequency components. Compared with LIF, TM-LIF, via channel-wise adaptive thresholds and multi-level quantization, yields spectra closer to the continuous responses of ReLU and alleviates the haze-induced loss of high-frequency details.

$$
U ( z ) = \beta z ^ { - 1 } U ( z ) + I ( z ) ,\tag{12}
$$

which gives the input-to-membrane transfer function

$$
G ( z ) \triangleq { \frac { U ( z ) } { I ( z ) } } = { \frac { 1 } { 1 - \beta z ^ { - 1 } } } .\tag{13}
$$

This corresponds to a first-order IIR filter with a pole at $z = \beta ;$ as $\beta  1$ , stronger low-pass smoothing suppresses high-frequency components, and stacking across layers accumulates into blurred structures and lost fine details.

To compensate for this compounded degradation under strict energy constraints, we propose SSM, which uses a lightweight Sobel-derived structural prior for structureguided channel and spatial modulation, selectively enhancing edge/texture responses in a spike-eficient manner. SSM is deployment-friendly: Sobel is fixed (parameter-free), and the modulation branch is lightweight (a compact MLP and a shallow spiking convolution), adding minimal overhead while preserving event-driven sparsity.

Sobel-derived Structural Prior. We instantiate the structural prior in SSM using fixed Sobel gradients. For each time step t, we first obtain a single-channel map $\hat { \mathbf { X } } _ { t }$ <sub>t</sub> by averaging $\mathbf { X } _ { t }$ along the channel dimension, and then compute the gradient-strength map:

$$
\mathbf { P } _ { t } = \big | \mathrm { S o b } _ { x } ( \hat { \mathbf { X } } _ { t } ) \big | + \big | \mathrm { S o b } _ { y } ( \hat { \mathbf { X } } _ { t } ) \big | ,\tag{14}
$$

where $\mathrm { S o b } _ { x } ( \cdot )$ and $\mathrm { S o b } _ { y } ( \cdot )$ denote fixed Sobel operators that compute the horizontal and vertical image gradients, respectively. To make this cue comparable across inputs and time steps, we further normalize it by its spatial mean:

$$
\hat { \mathbf { P } } _ { t } = \frac { \mathbf { P } _ { t } } { \operatorname { M e a n } _ { ( h , w ) } \mathopen { } \mathclose \bgroup \left( \mathbf { P } _ { t } \aftergroup \egroup \right) + \epsilon } .\tag{15}
$$

<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="2">LHID</td><td colspan="2">DHID</td><td colspan="2">RICE1</td><td colspan="2">RICE2</td><td colspan="2">Complexity</td></tr><tr><td>|PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>|PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>Param (M)</td><td>Energy (mJ)</td></tr><tr><td>DCP (He, Sun, and Tang 2010)</td><td>CVPR&#x27;09</td><td>15.86</td><td>0.7168</td><td>19.17</td><td>0.7561</td><td>16.03</td><td>0.6966</td><td>10.37</td><td>0.4491</td><td></td><td></td></tr><tr><td>DehazeNet (Cai et al. 2016)</td><td>TIP&#x27;16</td><td>15.91</td><td>0.7252</td><td>21.65</td><td>0.8389</td><td>20.94</td><td>0.8503</td><td>21.16</td><td>0.8412</td><td>0.008</td><td>2.48</td></tr><tr><td>FFA-Net (Qin et al. 2020)</td><td>AAAI&#x27;20</td><td>25.05</td><td>0.8800</td><td>27.79</td><td>0.8870</td><td>31.78</td><td>0.9543</td><td>26.81</td><td>0.7932</td><td>4.69</td><td>1322.64</td></tr><tr><td>4KDehazing (Zheng et al. 2021b)</td><td>CVPR&#x27;21</td><td>25.73</td><td>0.8406</td><td>23.44</td><td>0.8076</td><td>27.15</td><td>0.9062</td><td>26.72</td><td>0.9140</td><td>34.55</td><td>477.94</td></tr><tr><td>DCI-Net† (Zhang and Wang 2022)</td><td>TGRS&#x27;22</td><td>26.00</td><td>0.8930</td><td>28.12</td><td>0.9080</td><td>35.03</td><td>0.9600</td><td>32.14</td><td>0.8600</td><td>26.51</td><td>123.56</td></tr><tr><td>Dehamer (Guo et al. 2022)</td><td>CVPR&#x27;22</td><td>25.36</td><td>0.8486</td><td>26.15</td><td>0.8821</td><td>27.41</td><td>0.9059</td><td>26.73</td><td>0.8781</td><td>132.40</td><td>274.48</td></tr><tr><td>DehazeFormer-b (Song et al. 2023)</td><td>TIP&#x27;23</td><td>29.73</td><td>0.8964</td><td>24.98</td><td>0.8734</td><td>28.46</td><td>0.9053</td><td>27.61</td><td>0.8826</td><td>2.514</td><td>118.63</td></tr><tr><td>Trinity-Net† (Chi, Yuan, and Wang 2023)</td><td>TGRS&#x27;23</td><td>25.76</td><td>0.8780</td><td>24.78</td><td>0.8670</td><td>27.30</td><td>0.9000</td><td>24.71</td><td>0.7940</td><td>20.14</td><td>141.59</td></tr><tr><td>PCSFormer† (Zhang et al. 2024a)</td><td>TGRS&#x27;24</td><td>24.78</td><td>0.8375</td><td>28.68</td><td>0.8693</td><td>29.03</td><td>0.8894</td><td>26.99</td><td>0.8973</td><td>3.73</td><td>127.24</td></tr><tr><td>AU-Net† (Du et al. 2024)</td><td>TGRS&#x27;24</td><td>28.44</td><td>0.893</td><td>26.42</td><td>0.8910</td><td></td><td></td><td></td><td></td><td>7.14</td><td>538.43</td></tr><tr><td>SFSNiD (Cong et al. 2024)</td><td>CVPR&#x27;24</td><td></td><td></td><td>28.12</td><td>0.9030</td><td>35.93</td><td>0.9650</td><td>34.68</td><td>0.9080</td><td>8.35</td><td>319.05</td></tr><tr><td>DS-RDMPD† (Zhou et al. 2025)</td><td>TGRS&#x27;25</td><td>29.39</td><td>0.8854</td><td>27.91</td><td>0.9054</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SFRDP-Net† (Sun et al. 2025)</td><td>TGRS&#x27;25</td><td></td><td></td><td>28.89</td><td>0.9070</td><td>37.38</td><td>0.9650</td><td>35.45</td><td>0.9100</td><td>5.08</td><td>175.21</td></tr><tr><td>EM-SNN</td><td>Our</td><td>30.56</td><td>0.9106</td><td>28.83</td><td>0.9073</td><td>35.99</td><td>0.9630</td><td>36.25</td><td>0.9406</td><td>4.81</td><td>43.62</td></tr></table>

Table 1: Quantitative comparison on LHID, DHID, RICE1, and RICE2. Bold and underlined indicate the best and second-best results, respectively. <sup>†</sup> Methods specifically designed for remote sensing dehazing.

where $\hat { \mathbf { P } } _ { t } \in \mathbb { R } ^ { B \times 1 \times H \times W }$ denotes the structural prior that highlights edge- and texture-dominant regions and is leveraged to guide channel and spatial modulation within SSM. Channel-wise Modulation. The channel branch focuses on what feature channels should be emphasized or suppressed at each time step. At time step t, we compute three pooled descriptors over spatial dimensions $( h , w ) ;$

$$
\begin{array} { r l } & { \mathcal { D } _ { t } ^ { \mathrm { a v g } } = \mathrm { A v g P o o l } _ { ( h , w ) } ( \mathbf { X } _ { t } ) , } \\ & { \mathcal { D } _ { t } ^ { \mathrm { m a x } } = \mathrm { M a x P o o l } _ { ( h , w ) } ( \mathbf { X } _ { t } ) , } \\ & { \mathcal { D } _ { t } ^ { \mathrm { h f } } = \mathrm { A v g P o o l } _ { ( h , w ) } ( \mathbf { X } _ { t } \odot \hat { \mathbf { P } } _ { t } ) , } \end{array}\tag{16}
$$

where hf denotes a structural-prior-weighted descriptor, and $\mathcal { D } _ { t } ^ { \mathrm { a v g } } , \mathcal { D } _ { t } ^ { \mathrm { m a x } }$ , and $\mathcal { D } _ { t } ^ { \mathrm { h f } } \in \mathbb { R } ^ { B \times C \times 1 \times 1 }$ are the corresponding channel-wise descriptors. These descriptors are aggregated and fed into a shared MLP to generate the channel gate:

$$
\mathcal { G } _ { t } ^ { \mathrm { c h a } } = \mathrm { M L P } \Big ( \mathcal { D } _ { t } ^ { \mathrm { a v g } } + \mathcal { D } _ { t } ^ { \mathrm { m a x } } + \mathcal { D } _ { t } ^ { \mathrm { h f } } \Big ) ,\tag{17}
$$

where $\mathcal { G } _ { t } ^ { \mathrm { c h a } } ~ \in ~ \mathbb { R } ^ { B \times C \times 1 \times 1 }$ and the MLP parameters are shared across time steps. To retain identity initialization and allow both enhancement and suppression, we adopt residual re-weighting:

$$
\mathbf { X } _ { t } ^ { \mathrm { c h a } } = \mathbf { X } _ { t } + \mathbf { X } _ { t } \odot { \mathcal { G } } _ { t } ^ { \mathrm { c h a } } , \qquad t = 1 , \ldots , T ,\tag{18}
$$

Spatial-wise Modulation. While channel-wise modulation selects informative feature dimensions, the spatial branch focuses on where the enhancement should be applied. At each time step t, we form a spatial descriptor by concatenating per-pixel intensity statistics of $\mathbf { X } _ { t } ^ { \mathrm { c h a } }$ with the normalized structural prior. Then, this spatial descriptor is fed into a lightweight spiking convolution to predict a spatial gate $\mathcal { G } _ { t } ^ { s } \in \mathbb { R } ^ { B \times 1 \times H ^ { \star } W }$ , and obtain the Sobel-modulated representations:

$$
{ \bf X } _ { t } ^ { \prime } = { \bf X } _ { t } ^ { \mathrm { c h a } } + { \bf X } _ { t } ^ { \mathrm { c h a } } \odot \mathcal { G } _ { t } ^ { s } , \qquad t = 1 , \ldots , T ,\tag{19}
$$

The convolution operation is also shared across time steps to reduce parameters.

## Experiments

## Experimental Setup

Datasets. Following prior remote-sensing dehazing benchmarks (Sun et al. 2025; Zhu et al. 2025), we evaluate our method on four widely used datasets, including two synthetic datasets (HRSD and SateHaze1K) and two real-world datasets (RICE and RRSHID). HRSD (Zhang and Wang 2022) is a synthetic dataset consisting of LHID and DHID. LHID contains 30,517 training images and 500 test images synthesized from Google Earth imagery, while DHID contains 14,990 images in total, with 14,490 for training and 500 for testing; all images are resized to $5 1 2 \times 5 1 2$ . Sate-Haze1K (Huang et al. 2020) contains 1,200 paired hazy/- clear GF-2 images categorized into thin, moderate, and thick haze, and each subset is split into 320/35/45 images for training, validation, and testing. For real-world evaluation, RICE (Lin et al. 2019) consists of RICE1 with 500 Google Earth cloudy/clear pairs and RICE2 with 450 Landsat 8 OLI/TIRS image sets; both are randomly divided into training and testing sets at a ratio of9:1. RRSHID (Zhu et al. 2025) contains 3,053 paired hazy and haze-free images, which are further divided into thin, moderate, and thick haze subsets with 763, 1,526 and 764 pairs, respectively.

## Implementation Details.

All experiments are implemented in PyTorch and run on an NVIDIA RTX 4090 GPU. Unless stated otherwise, we set T=1 and D=4, with threshold scaling factor $\alpha { = } 0 . 6 ,$ leak factor $\beta { = } 0 . 2 5 .$ , and EMA momentum $\mu { = } 0 . 9 .$ . We optimize with AdamW $( \beta _ { 1 } { = } 0 . 9 , \ \beta _ { 2 } { = } 0 . 9 9 9 )$ using a cosineannealing learning-rate schedule, an initial learning rate of $1 \times 1 0 ^ { - 5 }$ , and a batch size of 4. During training, images are randomly cropped to $2 5 6 \times 2 5 6$ . The objective function is $\omega _ { 1 } \ell _ { 1 } + \omega _ { 2 } \ell _ { \mathrm { S S I M } }$ , with $\omega _ { 1 } { = } 1 . 0$ and $\omega _ { 2 } { = } 0 . 5 ,$ , and all models are trained for 1000 epochs. FLOPs and SOPs are reported at an input resolution of $2 5 6 \times 2 5 6$ . The scale-wise depth configuration is [4, 4, 8, 2, 2]. For back-propagation through spiking neurons, EM-SNN uses a rectangular surrogate gradient. Two widely acknowledged metrics, PSNR (Hore and Ziou 2010) and SSIM (Wang et al. 2004), are utilized to evaluate all datasets to assess EM-SNN.

## Experimental Results

Quantitative Results. Tables 1 to 3 report the PSNR and SSIM metrics for all methods. EM-SNN achieves SOTA performance on the LHID, RRSHID, and RICE2 datasets, compared with other ANN-based methods (Song et al. 2023;

![](images/c949635cf4ee0dbe6aee9900c1328d032a1455d09f9a8cc5c22d9f29dd50c72a.jpg)  
Figure 4: Qualitative comparisons on SateHaze1K under varying haze densities. The orange box marks the zoom-in region. The bottom rows show enlarged patches and absolute error maps w.r.t. the ground truth (darker blue indicates lower error). Our method recovers clearer structures and sharper boundaries with fewer residual haze artifacts and color distortions.

<table><tr><td rowspan="2">Method</td><td colspan="2">SateHaze1K-thin</td><td colspan="2">SateHaze1K-moderate</td><td colspan="2">SateHaze1K-thick</td><td colspan="2">|SateHaze1K-average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>DCP</td><td>20.47</td><td>0.8705</td><td>20.83</td><td>0.8971</td><td>16.71</td><td>0.7435</td><td>19.34</td><td>0.8370</td></tr><tr><td>DehazeNet</td><td>16.82</td><td>0.8210</td><td>15.75</td><td>0.7720</td><td>15.12</td><td>0.7430</td><td>15.90</td><td>0.7787</td></tr><tr><td>FFA-Net</td><td>25.16</td><td>0.9168</td><td>24.79</td><td>0.9296</td><td>22.36</td><td>0.8453</td><td>24.10</td><td>0.8972</td></tr><tr><td>4KDehazing</td><td>23.35</td><td>0.8939</td><td>23.73</td><td>0.9209</td><td>20.61</td><td>0.8227</td><td>22.56</td><td>0.8792</td></tr><tr><td>DCI-Net†</td><td>24.39</td><td>0.9022</td><td>25.03</td><td>0.9315</td><td>21.82</td><td>0.8390</td><td>23.75</td><td>0.8909</td></tr><tr><td>Dehamer</td><td>22.50</td><td>0.8973</td><td>25.73</td><td>0.9390</td><td>17.35</td><td>0.7957</td><td>21.86</td><td>0.8773</td></tr><tr><td>DehazeFormer-b</td><td>25.68</td><td>0.9193</td><td>27.15</td><td>0.9408</td><td>22.74</td><td>0.8494</td><td>25.19</td><td>0.9032</td></tr><tr><td>Trinity-Net†</td><td>23.46</td><td>0.8998</td><td>25.98</td><td>0.9372</td><td>20.47</td><td>0.8148</td><td>23.30</td><td>0.8839</td></tr><tr><td>PCSFormer†</td><td>22.08</td><td>0.8751</td><td>23.29</td><td>0.9365</td><td>18.39</td><td>0.7727</td><td>21.25</td><td>0.8614</td></tr><tr><td>OK-Net</td><td>22.16</td><td>0.8974</td><td>26.00</td><td>0.9437</td><td>20.51</td><td>0.8257</td><td>22.89</td><td>0.8889</td></tr><tr><td>AU-Net†</td><td>22.71</td><td>0.8992</td><td>21.04</td><td>0.9038</td><td>21.81</td><td>0.8458</td><td>21.85</td><td>0.8829</td></tr><tr><td>SFSNiD</td><td>25.90</td><td>0.9180</td><td>26.58</td><td>0.9370</td><td>24.35</td><td>0.8680</td><td>25.61</td><td>0.9077</td></tr><tr><td>DS-RDMPD†</td><td>26.03</td><td>0.9191</td><td>28.17</td><td>0.9444</td><td>22.97</td><td>0.8485</td><td>25.72</td><td>0.9040</td></tr><tr><td>EM-SNN</td><td>26.29</td><td>0.9239</td><td>27.43</td><td>0.9465</td><td>23.17</td><td>0.8575</td><td>25.63</td><td>0.9093</td></tr></table>

Table 2: Quantitative comparison on the SateHaze1K.

Zheng et al. 2021b). Furthermore, EM-SNN consistently surpasses the (Cong et al. 2024; Zhang and Wang 2022), achieving higher PSNR on both DHID (28.83 dB) and RICE1 (35.99 dB), respectively. From the haze-density perspective, EM-SNN performs best under thin haze, achieves the highest SSIM (0.9465) under moderate haze, and ranks second under thick haze in terms of both PSNR (23.17) and SSIM (0.8575), trailing only SFSNiD (Cong et al. 2024). These results demonstrate EM-SNN’s robust dehazing quality across datasets, which can be attributed to TM-LIF stabilizing spiking activations via variance-driven threshold modulation under haze-induced contrast compression, and SSM injecting structural priors to strengthen edge/texture recovery.

Qualitative Results. To clearly illustrate the dehazing capability of EM-SNN, we present qualitative comparisons on SateHaze1K in Fig. 4. The three rows in Fig. 4 correspond to the thin, moderate, and thick haze levels in SateHaze1K, respectively. Compared with other dehazing methods, EM-SNN restores clearer structures, preserves richer details, and produces fewer artifacts, which is consistent with the quantitative results.

Model Eficiency. To validate eficiency, we compare the energy consumption ofEM-SNN with prior methods. As shown in the last three columns of Tab. 1, in view of SNN eficiency is typically assessed in terms of energy, because computation is event-driven and depends on both dense MACs and spike operations (SOPs) (Luo et al. 2024). Following the energy estimation protocol in (Lei et al. 2025), we report the energy consumption in Tab. 4. Our model maintains low neural activity across datasets, leading to moderate spike operations and low energy consumption. The average energy consumption is 43.62 mJ. we set the operator energy constant to 4.6 pJ/MAC and use the FLOPs-to-energy-converted upper bound of SFRDP-Net (Sun et al. 2025) as the reference. Our method consumes only 25% of that upper bound.

<table><tr><td rowspan="2">Method</td><td colspan="2">RRSHID-thin</td><td colspan="2">RRSHID-moderate</td><td colspan="2">RRSHID-thick</td><td colspan="2">RRSHID-average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>|PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>DCP</td><td>18.46</td><td>0.4564</td><td>17.80</td><td>0.4856</td><td>18.39</td><td>0.4843</td><td>18.22</td><td>0.4754</td></tr><tr><td>FFA-Net</td><td>17.08</td><td>0.4452</td><td>17.40</td><td>0.5385</td><td>16.71</td><td>0.4792</td><td>17.06</td><td>0.4876</td></tr><tr><td>GridDehazeNet</td><td>22.77</td><td>0.6145</td><td>22.62</td><td>0.6468</td><td>23.96</td><td>0.7112</td><td>23.12</td><td>0.6575</td></tr><tr><td>4KDehazing</td><td>22.83</td><td>0.6177</td><td>22.47</td><td>0.6505</td><td>22.55</td><td>0.6912</td><td>22.62</td><td>0.6531</td></tr><tr><td>SCANet</td><td>18.37</td><td>0.4718</td><td>18.11</td><td>0.5380</td><td>19.07</td><td>0.5966</td><td>18.52</td><td>0.3741</td></tr><tr><td>Trinity-Net</td><td>20.51</td><td>0.5728</td><td>22.46</td><td>0.5728</td><td>24.11</td><td>0.7234</td><td>22.36</td><td>0.6230</td></tr><tr><td>DehazeFormer-b</td><td>22.74</td><td>0.6005</td><td>23.06</td><td>0.6137</td><td>24.69</td><td>0.7143</td><td>23.50</td><td>0.6428</td></tr><tr><td>PCSFormer</td><td>21.83</td><td>0.5427</td><td>22.09</td><td>0.5984</td><td>23.71</td><td>0.6547</td><td>22.54</td><td>0.5986</td></tr><tr><td>PhDNet</td><td>22.64</td><td>0.6054</td><td>22.92</td><td>0.6448</td><td>24.28</td><td>0.6996</td><td>23.28</td><td>0.6499</td></tr><tr><td>MCAF-Net</td><td>23.32</td><td>0.6236</td><td>23.60</td><td>0.6583</td><td>25.40</td><td>0.7221</td><td>24.11</td><td>0.6680</td></tr><tr><td>EM-SNN</td><td>25.10</td><td>0.7286</td><td>24.70</td><td>0.7115</td><td>26.02</td><td>0.7682</td><td>25.27</td><td>0.7361</td></tr></table>

Table 3: Performance comparison on RRSHID across haze densities. Results of baseline methods follow (Zhu et al. 2025).
<table><tr><td>Dataset</td><td>FR</td><td>SOPs (G)</td><td>Energy (mJ)</td><td>Dataset</td><td>FR</td><td>SOPs (G)</td><td>Energy (mJ)</td></tr><tr><td>SateHaze1K-thin</td><td>0.1729</td><td>34.938</td><td>46.166</td><td>DHID</td><td>0.2235</td><td>45.149</td><td>55.356</td></tr><tr><td>SateHaze1K-moderate</td><td>0.1688</td><td>34.110</td><td>45.421</td><td>LHID</td><td>0.1216</td><td>24.563</td><td>36.829</td></tr><tr><td>SateHaze1K-thick</td><td>0.1788</td><td>36.127</td><td>47.236</td><td>RRSHID-thin</td><td>0.1450</td><td>29.298</td><td>41.090</td></tr><tr><td>RICE1</td><td>0.1626</td><td>32.845</td><td>44.283</td><td>RRSHID-moderate</td><td>0.1322</td><td>26.709</td><td>38.761</td></tr><tr><td>RICE2</td><td>0.1547</td><td>31.257</td><td>42.854</td><td>RRSHID-thick</td><td>0.1291</td><td>26.078</td><td>38.193</td></tr></table>

Table 4: Energy evaluation of EM-SNN across diferent datasets. FR denotes the average firing rate.

<table><tr><td>ID</td><td>TM-LIF</td><td>SSM</td><td>PSNR↑</td><td>SSIM↑</td><td>Energy (mJ)↓</td></tr><tr><td>(1)</td><td></td><td></td><td>24.60</td><td>0.9019</td><td>47.76</td></tr><tr><td>(2)</td><td>√</td><td></td><td>24.93</td><td>0.9012</td><td>49.50</td></tr><tr><td>(3)</td><td></td><td>√</td><td>24.80</td><td>0.9033</td><td>48.36</td></tr><tr><td>(4)</td><td>√</td><td>√</td><td>25.63</td><td>0.9093</td><td>46.27</td></tr></table>

Table 5: Factorial ablation study of two modules on Sate-Haze1K.
<table><tr><td>ID</td><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>Energy (mJ)↓</td></tr><tr><td>(1)</td><td> $\mathrm { L I F } + \mathrm { S S M }$ </td><td>24.80</td><td>0.9033</td><td>48.36</td></tr><tr><td>(2)</td><td> $\mathrm { I - L I F + S S M }$ </td><td>25.00</td><td>0.9058</td><td>46.30</td></tr><tr><td>(3)</td><td> $\mathrm { N I - L I F } + \mathrm { S S M }$ </td><td>25.00</td><td>0.9081</td><td>37.26</td></tr><tr><td>(4)</td><td> $\mathrm { T M - L I F } + \mathrm { S S M }$ </td><td>25.63</td><td>0.9093</td><td>46.27</td></tr><tr><td>(5)</td><td> $\mathrm { T M - L I F } + \mathrm { w } / \mathrm { o } \mathrm { S S M }$ </td><td>24.93</td><td>0.9012</td><td>49.50</td></tr><tr><td>(6)</td><td> $\mathrm { T M \mathrm { - } L I F + S S M \left( f u l l \right) }$ </td><td>25.63</td><td>0.9093</td><td>46.27</td></tr><tr><td>(7)</td><td> $\mathrm { T M - L I F } + \mathrm { S S M } \mathrm { w } / \mathrm { o p r i o r }$ </td><td>25.30</td><td>0.9076</td><td>37.67</td></tr><tr><td>(8)</td><td> $\mathrm { T M \mathrm { - } L I F + S S M w / \hat { S c h a r r } p r i o r }$ </td><td>25.27</td><td>0.9079</td><td>38.81</td></tr></table>

Table 6: Additional ablations on SateHaze1K. The upper group compares neuron designs with SSM fixed, and the lower group compares SSM variants with TM-LIF fixed. Results are averaged over the thin/moderate/thick subsets.

## Ablation Study

For convenience, all ablation experiments are carried out on SateHaze1K and results are averaged over the thin/moderate/thick subsets. Additional experiments and ablation results are provided in the Supplementary Material.

Impact of Diferent Designs. The baseline is a spiking U-Net with standard LIF neurons $( v _ { \mathrm { t h } } { = } 0 . 1 5 )$ . TM-LIF replaces LIF with our threshold-modulated neuron, and SSM is added to the same backbone; Tab. 5 presents a factorial study of TM-LIF and SSM. Starting from 24.60 dB at 47.76 mJ, adding TM-LIF improves PSNR by +0.33 dB with an energy overhead of +1.73 mJ, while adding SSM yields +0.20 dB and +0.60 mJ. When both are activated, the gain increases to +1.03 dB (SSIM +0.0074) and energy is reduced by 1.49 mJ, indicating complementary efects and a better accuracy–eficiency trade-of.

Tab. 6 reports additional ablations on SateHaze1K. In the upper group, diferent neuron designs are compared with SSM fixed. TM-LIF achieves the best restoration performance, improving PSNR/SSIM from 24.80/0.9033 with standard LIF to 25.63/0.9093, while slightly reducing energy consumption from 48.36 to 46.27 mJ. Compared with TM-LIF, I-LIF (Luo et al. 2024) exhibits similar energy consumption but lower restoration quality, whereas NI-LIF (Lei et al. 2025) further reduces energy at the cost of a 0.63 dB PSNR drop. These results show that the statisticscalibrated channel-wise thresholds of TM-LIF provide a better accuracy–eficiency balance than fixed-threshold alternatives. In the lower group, diferent SSM variants are compared with TM-LIF fixed. Removing SSM decreases PSNR/SSIM from 25.63/0.9093 to 24.93/0.9012 and increases energy consumption from 46.27 to 49.50 mJ. Removing the structural prior or replacing Sobel with Scharr reduces energy consumption but also degrades restoration quality. Overall, the full Sobel-based SSM achieves the best restoration performance while maintaining moderate energy consumption. To visualize how firing thresholds vary with haze density, Figure 5 visualizes the frozen channel-wise firing thresholds learned by models separately trained on the thin, moderate, and thick haze subsets. Shallow layers generally exhibit lower and more concentrated thresholds, whereas deeper layers show broader distributions with partially higher values. Models trained on heavier haze exhibit an expanded threshold range and a more pronounced high threshold tail, indicating that the running statistics capture distribution-dependent shifts in membrane-potential scales rather than sample-wise adaptation during inference.

<table><tr><td colspan="3">T D PSNR ↑ SSIM↑ Energy (mJ) ↓</td></tr><tr><td>1 1 2</td><td>1 24.73 2 25.10 2 25.37</td><td>0.8998 0.9076 0.9072</td></tr><tr><td>1 4</td><td>25.63</td><td>65.19 0.9093 46.27</td></tr><tr><td>2 4</td><td>25.67</td><td>0.9116 80.19</td></tr><tr><td>4 4 1 8</td><td>25.81</td><td>0.9135 158.37</td></tr></table>

Table 7: Joint ablation of time steps T and spike depth D on SateHaze1K dataset.  
![](images/ea702e5ff4e6f9a2a292b07a38c03c4c950efb689141db5da6591cc3448c3f35.jpg)

![](images/3ad6e5b8c66895965cc993f130fb8530b3fb62e4d3b8fd5d9cb7b80343915a4e.jpg)  
(a) Thin

![](images/0e425fb128675506a4d07d8cd8205bc1206328a4319131c5a43ac2651f6a41ec.jpg)  
(b) Moderate  
(c) Thick  
Figure 5: Layer-wise distributions of frozen channel-wise thresholds $v _ { \mathrm { t h } }$ learned on (a) thin, (b) moderate, and (c) thick haze subsets (dark-to-light: shallow-to-deep layers).

Impact of Diferent Time Steps. We further study the impact of the number of time steps T and spike depth D on SateHaze1K (Tab. 7). Increasing T or D generally improves reconstruction but raises energy. The lightweight setting (T=1, D=1) achieves 24.73 dB/0.8998 at 30.41 mJ, while increasing to (T=1, D=4) yields +0.90 dB/+0.0095 at an energy overhead of +15.86 mJ, ofering a favorable accuracy–energy trade-of. Therefore we adopt T=1, D=4 by default.

## Conclusion

We proposed EM-SNN, a spiking neural network tailored for remote sensing image dehazing. EM-SNN combines a statistics-driven TM-LIF neuron with a SSM module to jointly counter haze-induced contrast compression and the discretization efects of spike generation, thereby improving structural preservation during deep spiking propagation. Experiments on public dehazing datasets demonstrate that EM-SNN achieves competitive restoration quality while maintaining sparse spiking activity and significantly reducing energy consumption. In the future, we will focus on enhancing the robustness of EM-SNN under more diverse real-world sensing conditions and extending its deployment to neuromorphic hardware platforms.

## References

Berman, D.; Avidan, S.; et al. 2016. Non-local image dehazing. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, 1674–1682.

Bi, G.; Si, G.; Zhao, Y.; Qi, B.; and Lv, H. 2021. Haze removal for a single remote sensing image using low-rank and sparse prior. IEEE Transactions on Geoscience and Remote Sensing, 60: 1–13.

Cai, B.; Xu, X.; Jia, K.; Qing, C.; and Tao, D. 2016. Dehazenet: An end-to-end system for single image haze removal. IEEE transactions on image processing, 25(11): 5187–5198.

Chi, K.; Yuan, Y.; and Wang, Q. 2023. Trinity-net: Gradientguided swin transformer-based remote sensing image dehazing and beyond. IEEE Transactions on Geoscience and Remote Sensing, 61: 1–14.

Cong, X.; Gui, J.; Zhang, J.; Hou, J.; and Shen, H. 2024. A semi-supervised nighttime dehazing baseline with spatialfrequency aware and realistic brightness constraint. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2631–2640.

Du, Y.; Li, J.; Sheng, Q.; Zhu, Y.; Wang, B.; and Ling, X. 2024. Dehazing network: Asymmetric unet based on physical model. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–12.

Fang, Y.; Zhou, D.; Wang, Z.; Ren, H.; Zeng, Z.; Li, L.; shibo zhou; and Xu, R. 2025. Spiking neural networks need highfrequency information. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Fattal, R. 2014. Dehazing using color-lines. ACM Transactions on Graphics, 34(1): 1–14.

Guo, C.-L.; Yan, Q.; Anwar, S.; Cong, R.; Ren, W.; and Li, C. 2022. Image Dehazing Transformer with Transmission-Aware 3D Position Embedding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Guo, Y.; Liu, X.; Chen, Y.; Zhang, L.; Peng, W.; Zhang, Y.; Huang, X.; and Ma, Z. 2023a. Rmp-loss: Regularizing membrane potential distribution for spiking neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 17391–17401.

Guo, Y.; Zhang, Y.; Chen, Y.; Peng, W.; Liu, X.; Zhang, L.; Huang, X.; and Ma, Z. 2023b. Membrane potential batch normalization for spiking neural networks. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 19420–19430.

He, K.; Sun, J.; and Tang, X. 2010. Single image haze removal using dark channel prior. IEEE transactions on pattern analysis and machine intelligence, 33(12): 2341– 2353.

Hore, A.; and Ziou, D. 2010. Image quality metrics: PSNR vs. SSIM. In 2010 20th international conference on pattern recognition, 2366–2369. IEEE.

Hu, S.; Ma, J.; Wan, J.; Min, W.; Jing, Y.; Zhang, L.; and Tao, D. 2025. ClusIR: Towards cluster-guided all-in-one image restoration. arXiv preprint arXiv:2512.10948.

Huang, B.; Zhi, L.; Yang, C.; Sun, F.; and Song, Y. 2020. Single satellite optical imagery dehazing using SAR image prior based on conditional generative adversarial networks. In Proceedings of the IEEE/CVF winter conference on applications ofcomputer vision, 1806–1813.

Jiang, N.; Li, H.-B.; Li, C.-J.; Xiao, H.-X.; and Zhou, J.- W. 2022. A fusion method using terrestrial laser scanning and unmanned aerial vehicle photogrammetry for landslide deformation monitoring under complex terrain conditions. IEEE Transactions on Geoscience and Remote Sensing, 60: 1–14.

Kim, S.; Park, S.; Na, B.; and Yoon, S. 2020. Spiking-yolo: spiking neural network for energy-eficient object detection. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 11270–11277.

Lee, D.; Li, Y.; Kim, Y.; Xiao, S.; and Panda, P. 2025. Spiking transformer with spatial-temporal attention. In Proceedings of the Computer Vision and Pattern Recognition Conference, 13948–13958.

Lee, J. H.; Delbruck, T.; and Pfeifer, M. 2016. Training deep spiking neural networks using backpropagation. Frontiers in neuroscience, 10: 508.

Lei, Z.; Yao, M.; Hu, J.; Luo, X.; Lu, Y.; Xu, B.; and Li, G. 2025. Spike2former: Eficient spiking transformer for high-performance image segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 1364–1372.

Li, B.; Peng, X.; Wang, Z.; Xu, J.; and Feng, D. 2017. Aodnet: All-in-one dehazing network. In Proceedings of the IEEE international conference on computer vision, 4770– 4778.

Li, J.; Hu, Q.; and Ai, M. 2018. Haze and thin cloud removal via sphere model improved dark channel prior. IEEE Geoscience and Remote Sensing Letters, 16(3): 472–476.

Liang, S.; Gao, T.; Chen, T.; and Cheng, P. 2024. A remote sensing image dehazing method based on heterogeneous priors. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–13.

Lin, D.; Xu, G.; Wang, X.; Wang, Y.; Sun, X.; and Fu, K. 2019. A remote sensing image dataset for cloud removal. arXiv preprint arXiv:1901.00600.

Liu, X.; Ma, Y.; Shi, Z.; and Chen, J. 2019. Griddehazenet: Attention-based multi-scale network for image dehazing. In Proceedings of the IEEE/CVF international conference on computer vision, 7314–7323.

Luo, X.; Yao, M.; Chou, Y.; Xu, B.; and Li, G. 2024. Integervalued training and spike-driven inference spiking neural network for high-performance and energy-eficient object detection. In European Conference on Computer Vision, 253–272. Springer.

Mladenova, I. E.; Bolten, J. D.; Crow, W. T.; Sazib, N.; Cosh, M. H.; Tucker, C. J.; and Reynolds, C. 2019. Evaluating the operational application of SMAP for global agricultural drought monitoring. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(9): 3387–3397.

Qin, X.; Wang, Z.; Bai, Y.; Xie, X.; and Jia, H. 2020. FFA-Net: Feature fusion attention network for single image dehazing. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 11908–11915.

Shi, X.; Hao, Z.; and Yu, Z. 2024. Spikingresformer: Bridging resnet and vision transformer in spiking neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5610–5619.

Song, T.; Jin, G.; Li, P.; Jiang, K.; Chen, X.; and Jin, J. 2024. Learning a Spiking Neural Network for Eficient Image Deraining. In Larson, K., ed., Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, 1254–1262. International Joint Conferences on Artificial Intelligence Organization. Main Track.

Song, Y.; He, Z.; Qian, H.; and Du, X. 2023. Vision transformers for single image dehazing. IEEE Transactions on Image Processing, 32: 1927–1941.

Sudevan, V.; Kausar, R.; Javed, S.; Karki, H.; De Masi, G.; and Dias, J. 2026. Spiking transformer with learnable threshold mechanism for underwater image dehazing to aid visionbased navigation. Neuromorphic Computing and Engineering.

Sun, H.; Yao, Z.; Du, B.; Wan, J.; Ren, D.; and Tong, L. 2025. Spatial–Frequency Residual-Guided Dynamic Perceptual Network for Remote Sensing Image Haze Removal. IEEE Transactions on Geoscience and Remote Sensing, 63: 1–16.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.

Xiao, Y.; Yuan, Q.; Jiang, K.; Huang, W.; Zhang, Q.; Zheng, T.; Lin, C.-W.; and Zhang, L. 2025. Spiking meets attention: Eficient remote sensing image super-resolution with attention spiking neural networks. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yao, M.; Richter, O.; Zhao, G.; Qiao, N.; Xing, Y.; Wang, D.; Hu, T.; Fang, W.; Demirci, T.; De Marchi, M.; et al. 2024. Spike-based dynamic computing with asynchronous sensing-computing neuromorphic chip. Nature Communications, 15(1): 4464.

Zhang, J.; Zhou, C.; Xu, K.; and Watanabe, M. 2002. Flood disaster monitoring and evaluation in China. Global Environmental Change Part B: Environmental Hazards, 4(2): 33–43.

Zhang, L.; and Wang, S. 2022. Dense haze removal based on dynamic collaborative inference learning for remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 60: 1–16.

Zhang, X.; Xie, F.; Ding, H.; Yan, S.; and Shi, Z. 2024a. Proxy and cross-stripes integration transformer for remote sensing image dehazing. IEEE Transactions on Geoscience and Remote Sensing.

Zhang, Y.; Luo, X.; Sun, Q.; Wang, Y.; Qu, H.; and Yi, Z. 2024b. Retina-Inspired lightweight spiking convolutional neural network for single-image dehazing. IEEE Transactions on Neural Networks and Learning Systems.

Zhao, K.; Hua, W.; Tuerhong, A.; Leng, L.; Ma, Y.; and Guo, Q. 2025. Threshold modulation for online test-time adaptation of spiking neural networks. arXiv preprint arXiv:2505.05375.

Zheng, H.; Wu, Y.; Deng, L.; Hu, Y.; and Li, G. 2021a. Going deeper with directly-trained larger spiking neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 35, 11062–11070.

Zheng, Z.; Ren, W.; Cao, X.; Hu, X.; Wang, T.; Song, F.; and Jia, X. 2021b. Ultra-high-definition image dehazing via multi-guided bilateral learning. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16180–16189. IEEE.

Zhou, H.; Wang, Y.; Zhang, Q.; Tao, T.; and Ren, W. 2025. A dual-stage residual difusion model with perceptual decoding for remote sensing image dehazing. IEEE Transactions on Geoscience and Remote Sensing, 63: 1–12.

Zhu, Q.; Mai, J.; and Shao, L. 2015. A fast single image haze removal algorithm using color attenuation prior. IEEE transactions on image processing, 24(11): 3522–3533.

Zhu, Z.-H.; Lu, W.; Chen, S.-B.; Ding, C. H. Q.; Tang, J.; and Luo, B. 2025. Real-world remote sensing image dehazing: benchmark and baseline. IEEE Transactions on Geoscience and Remote Sensing, 63: 1–14.