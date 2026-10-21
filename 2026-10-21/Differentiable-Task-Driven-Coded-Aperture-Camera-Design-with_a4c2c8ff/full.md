# Differentiable Task-Driven Coded Aperture Camera Design with 3D Gaussian Scene Representations

Chengyang Yan

Donald G. Dansereau

Australian Centre for Robotics, School of Aerospace, Mechanical and Mechatronic Engineering, The University of Sydney

chengyang.yan, donald.dansereau@sydney.edu.au

## Abstract

Camera design heavily influences perception task performance, yet cameras are typically designed by domain experts through costly experimentation, in isolation from the tasks they serve. Existing joint-design methods either rely on differentiable simulation limited to pre-captured 2D images, restricting them to optics or image-processing pipeline design, or use non-differentiable camera simulators optimised with derivative-free or reinforcementlearning methods that scale poorly with the number of design parameters. We present a differentiable framework for jointly designing complex camera systems and downstream perception tasks. We use it to design two codedaperture depth-from-defocus cameras, a monocular and a stereo configuration, by jointly optimising the coded aperture mask, camera parameters, and a depth estimation network end-to-end. Our framework represents scenes with 3D Gaussian Splatting reconstructed from real-world captures, providing a differentiable image formation process for gradient-based optimisation over design spaces intractable for prior differentiable approaches. Results demonstrate that our differentiable method improves over existing coded-aperture design and non-differentiable camera design baselines, offering a step forward in automated and principled camera designfor autonomous systems.

## 1. Introduction

The quality of camera captures directly governs the performance of downstream perception tasks, yet different autonomous systems require different camera designs to suit their tasks: autonomous vehicles need wide field of view (FoV) and high dynamic range, medical imaging needs fine resolution and colour fidelity, and robotics needs reliable depth and colour sensing. Hence, cameras should be designed in a task-specific manner.

Many works jointly optimise imaging systems and perception tasks via differentiable simulation and gradientbased optimisers [6, 8, 12, 16, 33, 37, 45]. However, their reliance on pre-captured 2D image datasets prevents generalisation to broader design problems such as FOV, resolution, multi-camera configurations, or unconventional architectures, limiting them to designing parts of the camera system (e.g., optics or the image signal processing pipeline) rather than full camera systems.

![](images/7fd6e551e798b40026936bb4033fd3074b0c4f1f0823ca2bc4ced0bba39f9a33.jpg)  
Figure 1. We introduce a differentiable method that jointly optimises camera parameters and perception tasks, using 3DGS scenes reconstructed from real-world captures and a coded aperture simulation modelling depth-dependent optical blur. Camera parameters and the perception network are jointly updated via gradientbased optimisation of the task loss (red arrows), spanning geometric parameters (FoV, pixel count, multi-camera configurations) and high-dimensional optical parameters (coded aperture mask).

Other works design full camera systems using 3D simulation environments, but treat the simulator as a non-differentiable black box, relying on reinforcementlearning [20] or derivative-free [35, 43, 44] optimisers that ignore gradient information and must optimise for long iterations. This becomes prohibitively inefficient as the number of design parameters grows, particularly for highdimensional parameters such as a coded aperture mask.

In this work, we propose a differentiable method for jointly designing complex camera systems and downstream perception tasks (Figure 1), using 3D Gaussian Splatting (3DGS) [18] for scene representation and differentiable image formation. This enables gradient-based optimisation of optics, sensors, and placement while retaining the capacity to model complex, unconventional camera systems beyond the reach of prior differentiable methods. Since 3DGS scenes can be reconstructed from real-world captures, our method trains on both rendered and captured data; discrete parameters such as the coded aperture mask and sensor selection are optimised via a continuous-relaxation surrogate gradient. We demonstrate scalability to high-dimensional design spaces by jointly optimising a coded aperture mask with the remaining camera parameters and a depth-fromdefocus perception task, for both single-camera and stereo configurations, analysing the resulting design trade-offs, an analysis made possible by our 3DGS-based representation and largely overlooked in prior coded aperture work. In summary, our contributions are:

• We introduce a differentiable framework that jointly optimises camera systems and their perception networks using 3DGS scenes reconstructed from real-world captures.

• We use this framework to design both monocular and stereo coded aperture cameras, via a depth-from-defocus network augmented with a frequency injection module and extended to the stereo setup.

• We systematically compare against existing camera design approaches and demonstrate significant improvements (e.g., up to 65% lower all-in-focus (AiF) LPIPS and 13% lower depth RMSE compared to the strongest baseline in the monocular setup), and analyse the design trade-offs between camera parameters, aperture masks, and task performance.

This work provides a unified, differentiable framework that bridges scene representation, image formation, and perception, enabling systematic study of how camera design choices affect downstream task performance.

## 2. Related Work

Task-Driven Camera Design Designing cameras for perception tasks improves performance, though many approaches rely on non-differentiable simulators. Early work used renderers such as ISET [39, 40] to prototype cameras and manually tune parameters against task outcomes [4, 23– 25, 41], trading laborious manual iteration for lower development cost. Later work automates this with reinforcementlearning [15, 20] or evolutionary [35, 43, 44] methods that bypass large networks for faster, more data-efficient optimisation. Treating the simulator as a black box, however, makes these methods inefficient with high-dimensional or numerous parameters such as a coded aperture mask, which our method handles efficiently via gradient-based optimisation while retaining full camera system support.

Another line of work instead leverages differentiable simulation to jointly optimise camera parameters and the perception task, applied to depth of field [33, 34, 46], object detection [8, 31, 36, 37], image classification [6, 9, 45, 49], depth estimation [1, 5, 14, 16], HDR imaging [26, 27], and motion deblurring [30]. These methods simulate cameras’ effect on pre-captured 2D images, however, limiting generalisation to variable FoV, resolution, unconventional configurations, or multi-camera systems. Our approach instead constructs the full environment in 3D using 3DGS, supporting such scenarios while retaining differentiability.

Coded Aperture Imaging Coded aperture imaging modifies the aperture to encode scene-dependent blur, improving depth and AiF recovery. Early work showed that appropriately designed aperture patterns improve the conditioning of depth-from-defocus and deblurring [21, 38, 50], and recent methods jointly optimise the aperture code and reconstruction network end-to-end using amplitude [32] or phase masks [13, 16, 42]. Ghanekar et al. [12] extended this to dual-pixel sensors, learning an amplitude mask that trades depth estimation against deblurring.

These methods optimise the aperture mask alone, whereas we treat coded aperture imaging as a broader codesign problem, jointly optimising the mask with other camera parameters for task-driven objectives. This enables systematic exploration of trade-offs such as depth accuracy, noise, and defocus blur, and we further extend coded aperture imaging to stereo-camera systems.

## 3. Methodology

We propose a differentiable framework that jointly optimises camera design and downstream perception tasks using a 3DGS scene representation, as illustrated in Figure 2.

## 3.1. Scene Representation and Image Formation

We represent the 3D scene using 3DGS [18], a differentiable splatting-based representation reconstructed before optimisation from a set of input images, which renders RGB images and depth maps via alpha-compositing. We assume a pinhole projection model, though other camera models could be used; this yields analytical gradients with respect to camera parameters, enabling gradient-based joint optimisation of intrinsics (e.g., focal length, sensor dimensions) and extrinsics (e.g., placement, orientation).

To simulate coded aperture defocus, we apply an occlusion-aware layered blur model adapted from Ikoma et al. [16]: the rendered scene is quantised into discrete depth layers, each convolved with a depth-dependent point spread function (PSF) determined by the coded mask, focus distance, and f-number, then composited back-to-front using the depth maps to account for occlusion between layers.

Following [43], we augment the rendered image with an affine noise model [11] combining signal-dependent Poisson noise and thermal noise calibrated on a physical camera. The calibrated noise parameters generalise to arbitrary exposure, gain, pixel size, and aperture size by scaling the effective gain with the ratios of pixel and aperture areas relative to the calibrated configuration, including the coded mask’s open-aperture fraction.

![](images/6a3464cf76cc08e41adc061a521bf033226887adfff81a75cc8b2f2bb562f9d4.jpg)  
Figure 2. A camera captures the real-world scene (blue arrow), which 3DGS models as a 3D Gaussian representation. At each step, this representation is rendered with the current geometric camera parameters into pinhole RGB images and depth maps; the coded aperture simulation then convolves these RGB renders with depth-dependent PSFs computed from the coded mask, focus distance, and f-numbe via a physical optics model, followed by a physics-based noise model. The resulting renders are fed to the perception task network, and the task loss $l _ { \mathrm { t a s k } }$ is backpropagated through both stages (red arrows) to jointly update all learnable parameters (green text).

## 3.2. Perception Task

Our framework is compatible with any differentiable perception task, but here we focus on an RGB-D task that jointly estimates depth from defocus blur and reconstructs an AiF image, as it depends directly on the optical parameters we optimise. Our monocular network builds on the depth-from-defocus architecture of Ikoma et al. [16], which deconvolves the coded aperture image into a focal stack via a Tikhonov inverse filter and predicts the AiF image and depth map from the stack and coded image using a U-Net backbone. We augment this backbone with a frequency injection module that extracts local defocus cues from patchwise Fourier spectra of the coded image and injects them at the U-Net bottleneck, improving depth separation.

To demonstrate our method’s ability to design beyond a single camera, we extend this network to a stereo setup: we deconvolve each view’s coded aperture image into a focal stack as before, then fuse them into intermediate AiF images via a differentiable Laplacian sharpness measure for photometric consistency across views. Inspired by [7], a frozen pretrained optical flow network, RAPIDFlow [29], estimates disparity from these images and warps the right focal stack into the left view’s coordinate frame. The left and warped right stacks, together with a disparity-derived depth cue, are then fed into the same task network to predict the left view’s AiF image and depth map. Illustration diagrams of both setups are in the supplementary material.

## 3.3. Optimisation

We formulate our framework as a joint optimisation problem over the camera parameters and the perception task, as illustrated in Figure 2. At each step, the 3DGS scene is rendered into RGB and depth images, passed through the coded aperture and noise simulation, and fed to the perception network; the resulting task loss $l _ { \mathrm { t a s k } }$ is backpropagated through the entire pipeline to jointly update all camera and network parameters using the Adam optimiser [19].

Camera Parameters We optimise the optics (focal length, f-number, coded aperture mask, focus distance) and image sensor (dimensions, pixel size) for the monocular setup, plus the baseline between cameras for the stereo setup. Because sparse view coverage near the scene can cause 3DGS renders to contain large black regions under wide FoVs, we apply a black-pixel regularisation loss $\mathcal { L } _ { \mathrm { b l k } } = \lambda _ { \mathrm { b l k } } \beta / \mathrm { f o c a l } .$ where $\beta$ is the rendered black-pixel fraction, focal is the optimisable focal length, and the weight $\lambda _ { \mathrm { b l k } } { = } 3 0$ is set empirically to match the scale of the other losses. This loss discourages the optimiser from exploiting short focal lengths where the reconstruction is degenerate; it therefore constrains the effective design space rather than expressing a preference of the downstream perception task, and reflects a limitation of the 3DGS representation’s view coverage.

Discrete Parameters The image sensor (chosen from a fixed catalogue for manufacturability) and the coded aperture mask are inherently discrete choices, which we optimise using a Gumbel-Softmax relaxation with a straightthrough estimator (STE) [17]. Given K candidates with logits $\phi _ { i }$ , the relaxed selection weights are

$$
y _ { i } = \frac { \exp ( ( \phi _ { i } + g _ { i } ) / \tau ) } { \sum _ { j = 1 } ^ { K } \exp ( ( \phi _ { j } + g _ { j } ) / \tau ) } , \quad g _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \mathrm { G u m b e l } ( 0 , 1 ) ,\tag{1}
$$

where the temperature $\tau$ is annealed towards zero during training so that y progressively sharpens towards a one-hot vector. The STE uses the hard selection arg max $( \phi _ { i } + g _ { i } )$ in the forward pass, so the pipeline always renders with a physically realisable discrete configuration, while gradients reach ϕ through the soft weights y in the backward pass.

Table 1. Quantitative results for monocular camera design, comparing our joint camera, mask, and perception optimisation against a fixedcamera baseline (Ikoma et al. [16], tested with their original Phase mask and with an Amp. (amplitude) mask matched to our settings) and a derivative-free joint optimiser (TaCOS [43]). Optimised parameters are marked •, fixed parameters •; our method outperforms both baselines under two noise scenarios.
<table><tr><td rowspan="2">Scenario</td><td rowspan="2">Method</td><td rowspan="2">Coded Mask M</td><td rowspan="2">Focus Dist.</td><td colspan="3">Camera Parameters f-num Focal Length</td><td rowspan="2">Pixel Size</td><td colspan="3">All-in-Focus</td><td colspan="3">Performance</td><td rowspan="2"></td></tr><tr><td>focus (m) f</td><td>focal (mm)</td><td>Sensor Size w × h (mm)</td><td>p (µm)</td><td>PSNR (dB) ↑</td><td>SSIM ↑ LPIPS ↓</td><td>RMSE (m) ↓</td><td>Depth MAE (m) ↓</td><td>log10 ↓ δ &lt; 1.25 ↑</td></tr><tr><td rowspan="4">Noise×100</td><td>Fixed Cam. + Phase + Ikoma et al. [16]</td><td></td><td>1.70 ●</td><td>1.8●</td><td>14.29 ●</td><td>14.47×9.94●</td><td>9.00 ●</td><td>22.587</td><td>0.669</td><td>0.308</td><td>0.131</td><td>0.080</td><td>0.102</td><td>0.910</td></tr><tr><td>Fixed Cam. + Amp. + Ikoma et al. [16]</td><td></td><td>1.70 ●</td><td>1.8 ●</td><td>14.29 ●</td><td>14.47×9.94●</td><td>9.00 ●</td><td>23.543</td><td>0.696</td><td>0.292</td><td>0.126</td><td>0.080</td><td>0.098</td><td>0.917</td></tr><tr><td>TaCOS + Amp. + Proposed Model</td><td>R</td><td>0.99</td><td>11.9●</td><td>24.95</td><td>14.44×9.90</td><td>4.50</td><td>24.494</td><td>0.767</td><td>0.401</td><td>0.169</td><td>0.115</td><td>0.131</td><td>0.868</td></tr><tr><td>Ours - Optim. Cam. + Amp. + Proposed Model</td><td></td><td>0.45</td><td>12.4●</td><td>12.37</td><td>14.47×9.94</td><td>9.00</td><td>29.301</td><td>0.893</td><td>0.102</td><td>0.110</td><td>0.070</td><td>0.091</td><td>0.938</td></tr><tr><td rowspan="4">Noise×200</td><td>Fixed Cam. + Phase + Ikoma et al. [16]</td><td></td><td>1.70 ●</td><td>1.8●</td><td>14.29●</td><td>14.47×9.94●</td><td>9.00 ●</td><td>22.325</td><td>0.668</td><td>0.318</td><td>0.133</td><td>0.078</td><td>0.102</td><td>0.912</td></tr><tr><td>Fixed Cam. + Amp. + Ikoma et al. [16]</td><td>OO</td><td>1.70●</td><td>1.8 ●</td><td>14.29 ●</td><td>14.47×9.94●</td><td>9.00 ●</td><td>23.531</td><td>0.700</td><td>0.289</td><td>0.128</td><td>0.082</td><td>0.103</td><td>0.912</td></tr><tr><td>TaCOS + Amp. + Proposed Model</td><td>4</td><td>1.03●</td><td>9.0</td><td>24.07</td><td>14.13×7.45</td><td>3.45</td><td>23.644</td><td>0.772</td><td>0.436</td><td>0.182</td><td>0.127</td><td>0.136</td><td>0.850</td></tr><tr><td>Ours - Optim. Cam. + Amp. + Proposed Model</td><td></td><td>0.47</td><td>11.0●</td><td>14.60</td><td>14.47×9.94</td><td>9.00</td><td>27.534</td><td>0.872</td><td>0.116</td><td>0.117</td><td>0.077</td><td>0.099</td><td>0.932</td></tr></table>

For sensor selection, K is the catalogue size, and following [43] we optimise the underlying continuous sensor parameters θ (pixel size, sensor width, and height) rather than ϕ directly, setting ϕ to the negative normalised squared distance between θ and catalogue entry $\mathbf { c } _ { i } .$ This makes the nearest catalogue entry the most likely selection and implicitly pulls θ towards the catalogue during optimisation, avoiding the need for an explicit commitment loss. For the coded aperture mask, we apply the same formulation independently to each element with K = 2 (closed/open).

Perception Task We supervise the predicted AiF image <sup>ˆ</sup>I and depth map $\hat { D }$ against ground truth I and D rendered by 3DGS, using a combination of pixel-wise, structural, and colour losses. As I and $D$ are 3DGS renders rather than independent measurements, they may carry the reconstruction’s own error; since every method shares the same reference, the resulting metrics remain valid for comparing methods rather than as absolute measures of real-world accuracy. For the monocular network,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { m o n o } } = \lambda _ { \mathrm { a i f } } \big ( \| \hat { I } - I \| _ { 1 } + \lambda _ { \mathrm { g } } \mathcal { L } _ { \mathrm { e d g e } } ( \hat { I } , I ) \big ) } \\ & { \quad \quad + \lambda _ { \mathrm { d } } \big ( \| \hat { D } - D \| _ { 1 } + \lambda _ { \mathrm { g } } \mathcal { L } _ { \mathrm { e d g e } } ( \hat { D } , D ) \big ) } \\ & { \quad \quad + \lambda _ { \mathrm { s s i m } } \big ( \mathcal { L } _ { \mathrm { s s i m } } ( \hat { I } , I ) + \mathcal { L } _ { \mathrm { s s i m } } ( \hat { D } , D ) \big ) } \\ & { \quad \quad + \lambda _ { \mathrm { c h } } \mathcal { L } _ { \mathrm { c h } } + \lambda _ { \mathrm { c s } } \mathcal { L } _ { \mathrm { c s } } , } \end{array}\tag{2}
$$

where $\mathcal { L } _ { \mathrm { e d g e } }$ is a Sobel-gradient L1 loss that sharpens edges, $\mathcal { L } _ { \mathrm { s s i m } } ( \hat { y } , y \bar { ) } = 1 - \mathrm { S S I M } ( \hat { y } , y ) , \mathcal { L } _ { \mathrm { c h } }$ is an L1 loss on the CIE-Lab $a ^ { * } b ^ { * }$ channels penalising hue/saturation errors, and ${ \mathcal { L } } _ { \mathrm { c s } }$ matches per-channel RGB mean and standard deviation to correct colour casts and contrast mismatches. We set $( \lambda _ { \mathrm { a i f } } , \lambda _ { \mathrm { d } } , \lambda _ { \mathrm { s s i m } } , \lambda _ { \mathrm { g } } , \lambda _ { \mathrm { c h } } , \lambda _ { \mathrm { c s } } ) \ = \ ( 1 , 2 , 1 , 1 , 0 . 1 , 0 . 1 )$ , with the larger depth weight reflecting the greater difficulty of metric depth estimation relative to RGB reconstruction.

For the stereo network, we add a loss between the ground-truth depth and the depth $\bar { D }$ implied by the predicted disparity, giving

$$
\mathcal { L } _ { \mathrm { s t e r e o } } = \mathcal { L } _ { \mathrm { m o n o } } + \| \bar { D } - D \| _ { 1 } ,\tag{3}
$$

which also lets gradients optimise the baseline between cameras for disparity accuracy even though the flow network itself is frozen.

Total Loss The total loss for the framework is $\mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } +$ $\mathcal { L } _ { \mathrm { b l k } }$ , where $\mathcal { L } _ { \mathrm { t a s k } }$ is the monocular or the stereo loss.

## 4. Experiments

We validate our method on two close-range design scenarios for depth-from-defocus and AiF reconstruction, focusing on short range because the defocus cue is less informative further out. The monocular case targets robotic grasping, recovering depth from a single camera over a confined workspace. The stereo case targets macro photography, where a zoomed-in view with a shallow depth of field (DoF) produces strong defocus blur; here a stereo pair adds complementary photometric and geometric depth cues.

## 4.1. Implementations

Dataset and Environment We select close-range scenes from MipNeRF 360 [2] and LLFF [28]: 6 scenes (leaves, flower, fortress, orchids, bonsai, room) for the monocular case, and 7 scenes (the same 4 LLFF scenes plus counter, kitchen, bonsai from MipNeRF 360) for the stereo case. For each scene, we establish a 3D Gaussian representation with gsplat [47] as the virtual environment, then recover metric scale by aligning the reconstruction to metric depth from Depth Anything 3 [22]. The camera design and task network are jointly optimised across all scenes, with a subset of views per scene withheld for evaluation.

![](images/35a56a8f64732fd0e48140aa9610d8fec1c17dcfa6c6391e266114255b82da0e.jpg)  
Figure 3. Qualitative comparison for the monocular experiment. Our jointly optimised camera, mask, and perception model produce sharp AiF images and accurate depth, while the fixed-camera baseline (Ikoma et al. [16]) and the derivative-free joint optimiser (TaCOS [43]) degrade noticeably in blurry regions.

Noise Calibration We calibrate the noise model on a Basler Dart DaA1280-54uc camera [3] (5 ms exposure, 10 dB gain, f/8) following [43]. This configuration is treated as the reference point for generalisation.

Optimisation All camera parameters, coded aperture masks, and task networks are jointly optimised with Adam [19] for 20,000 steps with a batch size of 3, using a learning rate of $1 0 ^ { - 2 }$ for the camera parameters and $1 0 ^ { - \bar { 3 } }$ for the mask and task networks.

## 4.2. Monocular Camera

The monocular camera design task focuses on close-range RGB-D imaging from a single capture, which is a common requirement in robotics applications such as grasping.

Design Space The design space covers the optical parameters (focal length $\in [ 5 , 2 5 ]$ ] mm, focus distance $\in [ 0 . 1 , 5 ] \mathrm { m } .$ f-number $\in \ [ 1 . 4 , 1 6 ]$ , and a $1 7 \times 1 7 \mathrm { p x }$ circular amplitude coded aperture mask) and the image sensor, chosen from a catalogue of 43 commercial CMOS sensors collected in [43]. Sensor size and focal length determine the FoV, while focal length and f-number determine the aperture size. The mask and sensor choice are optimised with the Gumbel-Softmax and STE formulation, as in Section 3.3.

Comparison We compare against TaCOS [43], a genetic algorithm-based camera design method that jointly optimises camera parameters and the perception model, run for the same ${ \sim } 2 0 { , } 0 0 0$ -step budget (590 candidates over 34 generations, following its recommended 2×-parameter population size for our 295 optimised parameters).

We also compare to the baseline depth-from-defocus method of Ikoma et al. [16] with a fixed camera, testing both their original phase mask, matched in element count, and an amplitude mask matched to our own settings. We use the catalogue’s largest-pixel sensor for the best SNR, and reuse their focus distance as it targets a similar depth range to our scenes. We scale the f-number and focal length to reproduce their aperture diameter, matching their level of defocus blur while keeping a reasonable FoV with our sensor. These camera parameters are listed in Table 1.

To highlight the benefit of a coded aperture, which recovers an AiF image from the blur caused by widening the aperture under low-light conditions, we evaluate all methods under two low-SNR scenarios with increased image noise, representing low-light operation or noisier sensors.

AiF quality is measured with PSNR, SSIM, and the perceptual metric LPIPS [48]; depth accuracy is measured with RMSE, MAE, $\log _ { 1 0 }$ error, and the threshold accuracy $\delta \ < \ 1 . 2 5$ that measures the fraction of predicted depths within a factor of 1.25 of the ground truth.

Results Table 1 reports the camera parameters, coded aperture masks, and metrics for each method under both scenarios, with qualitative results for the Noise×100 scenario in Figure 3 and Noise×200 in the supplementary material.

Ikoma et al. [16]’s fixed camera has a large aperture diameter, producing strong defocus blur across the scene depth range that severely limits AiF quality, as the task model must invert a poorly conditioned PSF with little residual texture. The amplitude mask compensates by concentrating transmittance near the aperture centre to shrink the PSF spread, while keeping sparse coding at the boundary to retain depth cues. AiF quality suffers, but depth metrics stay competitive since the large aperture still yields strong defocus cues. Under higher noise, both variants degrade only marginally, as the mask opens more pixels to maintain light throughput.

Table 2. Quantitative results for stereo camera design. The monocular row is our own strongest monocular configuration evaluated under the same narrow FoV, isolating the benefit of a second view; CADS [12] is the closest existing coded-aperture baseline, included despit its dual-pixel baseline being limited to the sub-mm aperture scale. Optimised parameters are marked •, fixed •; stereo values are given as Left/Right (CADS shares parameters across views). Our stereo method outperforms both, especially in AiF reconstruction.
<table><tr><td rowspan="2">Scenario</td><td rowspan="2">Method</td><td colspan="3">Camera Parameters Coded Mask Focus Dist.</td><td rowspan="2">Baseline</td><td colspan="7">Performance All-in-Focus</td></tr><tr><td>focus (m)</td><td>f-num</td><td>b (m)</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>RMSE (m) ↓</td><td>MAE (m) ↓</td><td>Depth log10 ↓</td><td>δ &lt; 1.25 ↑</td></tr><tr><td rowspan="3">Noise×100</td><td>Fixed Cam. + CADS [12]</td><td></td><td>0.29 ●</td><td>9.87 ●</td><td>NA</td><td>26.397</td><td>0.759</td><td>0.301</td><td>0.244</td><td>0.135</td><td>0.188</td><td>0.797</td></tr><tr><td>Ours - Monocular</td><td></td><td>0.29 </td><td>16.0 ●</td><td>NA</td><td>26.335</td><td>0.819</td><td>0.205</td><td>0.089</td><td>0.058</td><td>0.114</td><td>0.912</td></tr><tr><td>Ours - Stereo</td><td>7 ●</td><td>0.26/0.89</td><td>16.0/11.3</td><td>0.01●</td><td>26.828</td><td>0.841</td><td>0.161</td><td>0.084</td><td></td><td>0.049 0.086</td><td>0.933</td></tr><tr><td rowspan="3">Noise×200</td><td>Fixed Cam. + CADS [12]</td><td></td><td>0.28 ●</td><td>9.87 ●</td><td>NA</td><td>25.618</td><td>0.718</td><td>0.350</td><td>0.259</td><td>0.143</td><td>0.203</td><td>0.775</td></tr><tr><td>Ours - Monocular</td><td></td><td>0.28</td><td>16.0</td><td>NA</td><td>25.678</td><td>0.788</td><td>0.239</td><td>0.089</td><td>0.051</td><td>0.115</td><td>0.913</td></tr><tr><td>Ours - Stereo</td><td></td><td>0.26/0.86</td><td>16.0/11.1●</td><td>0.01●</td><td>26.268</td><td>0.813</td><td>0.203</td><td>0.081</td><td>0.051</td><td>0.093</td><td>0.928</td></tr></table>

![](images/72e8a0037a8bdc5053a7aef7957b640fee3cc511d54e2aeabb1087dc9e6c02f8.jpg)  
Figure 4. Our method fuses photometric cues from a foreground-focused and a background-focused view for higher AiF quality and improves depth accuracy by leveraging disparity. Full comparisons with baselines are presented in the supplementary material.

TaCOS [43] selects a long focal length and high fnumber. This reduces defocus blur relative to Ikoma et al. [16] and modestly improves AiF PSNR/SSIM, but LPIPS and depth accuracy remain substantially worse than others. We attribute this to two factors: TaCOS represents the mask as a 1D array, preventing it from learning 2D spatial coding patterns and hurting LPIPS in particular, and its derivative-free optimiser lacks gradient information to resolve interactions among the many camera parameters. Under Noise×200, it lowers the f-number to widen the aperture, but also selects a smaller pixel size sensor that offsets the SNR gain, indicating that TaCOS struggles to find globally consistent configurations at this parameter scale, though more generations may help. It also takes substantially longer to optimise (3.6 h vs. our 2.1 h on a NVIDIA RTX PRO 6000 GPU) due to the smaller pixel sizes it selects and its optimisation method.

Our method converges to a short focal length and high f-number, giving a small aperture, ≈1.0 mm under Noise×100, and correspondingly large DoF that keeps the PSF well-conditioned across the workspace, enabling accurate AiF reconstruction while retaining enough residual defocus for the best depth results across all metrics. It also selects the largest available pixel size to maximise SNR at this small aperture, and the mask compensates for the small aperture area by opening more pixels to maintain light throughput. Under Noise×200, it modestly increases the focal length and lowers the f-number, enlarging the aperture to ≈1.3 mm to admit more light while keeping the same sensor. Despite the resulting increase in blur, it sustains strong performance and outperforms all baselines by a substantial margin in both scenarios. Note that the black-pixel regularisation loss discourages the optimiser from selecting even shorter focal lengths, so the converged values reported here reflect this constrained design space rather than an unconstrained preference of the perception task.

## 4.3. Stereo Camera

The stereo experiment targets a close-range, small-FoV macro-photography setting, aiming for a camera pair whose complementary photometric and geometric cues improve on the monocular baseline.

Design Space We fix the focal length (25 mm) and sensor (w=14.47, h=9.94 mm, $_ { p = 9 . 0 0 \mu \mathrm { m } }$ , from the monocular catalogue) identically for both cameras, giving a zoomedin, high-SNR view and the matched FoV needed for reliable disparity estimation. Each camera independently optimises its focus distance (∈ [0.1, 5] m), f-number (∈ [1.4, 16]), and mask as in the monocular setup, plus a shared stereo baseline $( b \in [ 0 . 0 1 , 1 . 0 0 ]$ m).

Table 3. Ablation study under Noise×100. Monocular: a fixed human-designed mask [21], a fixed camera (Ikoma et al. [16] configuration), and no frequency injection. Stereo: shared camera parameters, a shared mask, a conventional (uncoded) circular aperture, and no rightview warping. Optimised parameters are marked •, fixed $\bullet ; \bullet \bullet$ marks a parameter not applicable to that scenario. The results confirm that each component or design choice is a meaningful contributor to the proposed design.
<table><tr><td rowspan="2">Scenario</td><td rowspan="2">Method</td><td rowspan="2">Coded Mask</td><td colspan="4">Camera Parameters Focus Dist. f-num Focal Length</td><td rowspan="2">Pixel Size Baseline</td><td colspan="7">All-in-Focus</td></tr><tr><td>focus (m) f</td><td></td><td>Sensor Size focal (mm) w × h (mm)</td><td>p (µm)</td><td>b (m)</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>RMSE (m) ↓</td><td>Depth MAE (m) ↓ log10 ↓</td><td>δ &lt; 1.25 ↑</td></tr><tr><td rowspan="4">Monocular</td><td>Fixed Mask</td><td></td><td>0.56</td><td>5.0●</td><td>12.96</td><td>14.47×9.94</td><td>9.00</td><td></td><td>27.575</td><td>0.838</td><td>0.153</td><td>0.124</td><td>0.079 0.098</td><td>0.925</td></tr><tr><td>Fixed Camera</td><td></td><td>1.70 ●</td><td>1.8●</td><td>14.29●</td><td>14.47×9.94</td><td>9.00●</td><td></td><td>24.397</td><td>0.740 0.290</td><td>0.124</td><td>0.081</td><td>0.095</td><td>0.920</td></tr><tr><td>No Freq.</td><td>， L.</td><td>0.45</td><td>12.4</td><td>14.87</td><td>14.47×9.94</td><td>9.00</td><td></td><td>29.123</td><td>0.895 0.100</td><td>0.125</td><td>0.081</td><td>0.101</td><td>0.925</td></tr><tr><td>Proposed</td><td></td><td>0.45</td><td>12.4</td><td>12.37</td><td>14.47×9.94</td><td>9.00</td><td></td><td>29.301</td><td>0.893 0.102</td><td></td><td>0.110</td><td>0.070 0.091</td><td>0.938</td></tr><tr><td rowspan="5">Stereo</td><td>Shared Camera</td><td></td><td>0.29/0.29</td><td>16.0/16.0</td><td>25.00 ●</td><td>14.47×9.94●</td><td>9.00●</td><td>0.01 26.123</td><td>0.815</td><td>0.223</td><td>0.101</td><td>0.055</td><td>0.099</td><td>0.913</td></tr><tr><td>Shared Mask</td><td></td><td>0.26/0.83</td><td>16.0/10.0</td><td>25.00●</td><td>14.47×9.94●</td><td>9.00●</td><td>0.01</td><td>26.662 0.832</td><td>0.177</td><td>0.083</td><td>0.049</td><td>0.089</td><td>0.931</td></tr><tr><td>Circular Aperture</td><td></td><td>0.27/0.94</td><td>16.0/11.5</td><td>25.00●</td><td>14.47×9.94●</td><td>9.00●</td><td>0.01</td><td>25.423 0.781</td><td>0.236</td><td>0.091</td><td>0.056</td><td>0.091</td><td>0.921</td></tr><tr><td>w/o Warping</td><td>3 e</td><td>0.27/1.48</td><td>16.0/7.13</td><td>25.00●</td><td>14.47×9.94●</td><td>9.00●</td><td>0.01</td><td>26.567</td><td>0.826 0.205</td><td>0.084</td><td>0.051</td><td>0.093</td><td>0.926</td></tr><tr><td>Proposed</td><td>0.</td><td>0.26/0.89</td><td>16.0/11.3 1</td><td>25.00 ●</td><td>14.47×9.94●</td><td>9.00●</td><td>0.01</td><td>26.828</td><td>0.841</td><td>0.161</td><td>0.084</td><td>0.049 0.086</td><td>0.933</td></tr></table>

Comparison We compare against our own monocular design (under the same FoV), the strongest method from the monocular experiment, and against CADS [12], a dualpixel method that jointly learns a coded mask with a depthestimation task. A dual-pixel camera is conceptually similar to a stereo pair in that it captures two views from slightly different perspectives by splitting the aperture into two halves, each projecting onto a separate photodiode within each pixel, but its effective baseline is limited to the aperture diameter rather than a physical camera separation. We match its focal length, sensor, and aperture diameter (via fnumber) to our setup, but use our optimised focus distance in place of its own, as the original setting targets an extremely shallow depth range unsuitable for our scenes. All experiments use the same low-SNR scenarios and metrics as the monocular study.

Results Table 2 and Figure 4 report the results; full qualitative comparisons are in the supplementary material.

Our method converges to a complementary two-view design: the left camera takes the maximum f-number $\left( f \mathrm { = } 1 6 . 0 \right.$ , aperture ≈1.56 mm) and a short focus distance (0.26 m), maximising DoF for the sharpest possible AiF base images, while the right camera focuses further out (0.89 m) at a lower f-number (≈11.3, aperture ≈2.2 mm), adding depth cues from a different focal plane. The optimised baseline is very small (b=10 mm), which is enough for accurate right-to-left warping and photometric fusion, but not larger, since defocus already provides strong geometric depth cues at this range. This complementary design consistently improves on the monocular baseline, most notably in perceptual quality and depth accuracy. The masks follow the same logic: the left mask mirrors the monocular one (small central opening, sparse coded periphery, since it carries the primary supervision signal), while the right mask opens more, and asymmetrically towards the side away from the left camera. This is the region occluded from the left view, where the optimiser raises aperture weight to capture complementary information. Under Noise×200, both masks open further to compensate for SNR, and the stereo method retains its AiF and depth advantage over monocular. The MAE gain vanishes, however, as the extra noise degrades the deconvolved focal stack feeding the disparity estimator, weakening the right-to-left warp and depth prior.

CADS [12] matches our monocular AiF PSNR but is markedly worse in perceptual quality, indicating it preserves mean intensity at the expense of texture. This also shows that our Tikhonov-deconvolution step aids AiF reconstruction. Its depth accuracy is considerably weaker, which we attribute to a depth-range mismatch: CADS is trained on defocus size for depth, and targets millimetre-scale scenes (e.g. endoscopy) where defocus size varies monotonically with depth. Our 0.1–5 m range instead makes this relationship non-linear and ambiguous near the focus plane, so supervising on defocus size no longer yields accurate depth, and its dual-pixel baseline is also too small to compensate geometrically. Its mask reflects this: a large open centre with sparse peripheral coding that splits the aperture to encode disparity rather than modulating spatial frequency, limiting its coding richness relative to our masks that are free across the full aperture area.

![](images/d8da8b043ad515d8cdb94ef575cc66693b0b1a266fa03e7fbdc72df015e6afe9.jpg)  
Figure 5. Prototype of the optical system used for real-world validation and the real-world stereo results.

## 4.4. Ablation Study

We isolate the contribution of key design choices in both experiments under Noise×100. Results are in Table 3.

For the monocular method, fixing the mask to the human-designed pattern of Levin et al. [21] makes the camera optimiser widen the aperture to compensate for its lower open fraction. Performance still slightly degrades across all metrics, however, demonstrating that optimising camera parameters alone cannot recover the spatial structure of a jointly optimised 2D mask. Fixing the camera to the Ikoma et al. [16] configuration substantially reduces AiF quality, confirming that its large aperture and shallow DoF limit reconstruction. It still outperforms the equivalent fixedcamera baseline from the main comparison, however, showing that our model and losses help even without camera optimisation. Removing the frequency injection module (reverting to the Ikoma et al. backbone) leaves AiF quality largely unchanged but degrades depth accuracy. The optimiser compensates with a longer focal length to increase blur, confirming the module mainly aids depth estimation via depth-dependent defocus features.

For the stereo method, sharing camera parameters across views collapses both to the left-camera design, removing the complementary focus split and degrading both AiF and depth. The right-view mask compensates by becoming horizontally anisotropic, encoding directional defocus cues since it can no longer differentiate through focus or aperture. Sharing a mask across views causes a modest AiF drop (LPIPS 0.177 vs. 0.161) but slightly improves depth (RMSE 0.083 vs. 0.084), as identical defocus patterns give more photometrically consistent input to the disparity estimator. Replacing the coded mask with a conventional open aperture, reducing to a standard stereo camera, significantly degrades AiF reconstruction (LPIPS 0.236 vs. 0.161): an uncoded circular aperture produces a PSF that is poorly conditioned for deconvolution, so sharpness cannot be recovered through reconstruction the way it can with the coded pattern. Depth accuracy degrades mildly, as stereo disparity still recovers depth correctly over most of the scene, though the weaker AiF reconstruction locally blurs the depth prediction near edges. Removing the warping stage pushes the right camera to a much longer focus distance and wider aperture, creating stronger and more spatially distinctive blur that can be exploited without requiring pixel-level correspondence. This only mildly hurts depth accuracy but degrades AiF (LPIPS 0.205 vs. 0.161). The mask also opens more on the side overlapping the left camera’s FoV, the region of highest cross-view photometric consistency it can still exploit without explicit correspondence.

## 4.5. Real-World Validation

We further validate our method on a prototype built from a commercial FLIR Flea3 camera [10] with a C-mount lens (Figure 5) due to manufacturing constraints. Hence, this validation focuses on optimising the coded mask, focus distance, and perception network, with the remaining optics fixed. The test scene’s 3DGS representation is added to the training set, though the evaluated view itself is not part of the training views. We validate both the monocular and stereo settings, with setup details, PSF calibrations, and monocular results reported in the supplementary material. The stereo camera is achieved by moving the camera by the baseline distance, and the corresponding results are shown in Figure 5. The real-world behaviour matches our simulation: the stereo method focuses each view at a different depth to fuse complementary cues for AiF reconstruction, and the recovered depth aligns with the true scene geometry, confirming the method’s applicability beyond simulation.

## 5. Conclusion

We presented a differentiable framework that jointly optimises camera design and downstream perception tasks, using 3D Gaussian representations reconstructed from real captures as a differentiable virtual environment for image formation. This avoids the scalability limits of derivativefree optimisation. We validated the framework on monocular and stereo depth-from-defocus tasks with jointly optimised coded apertures, sensors, and inter-camera geometry. It consistently outperforms fixed-camera and derivativefree baselines, while revealing interpretable design tradeoffs that a decoupled process would miss. We believe this is an important step toward principled, automated camera design for autonomous systems that accounts for the interdependency between cameras and the algorithms that interpret them. For future work, we aim to extend the framework to jointly optimise camera count and array topology, and to incorporate adaptive, task-driven control of camera parameters online at inference time.

## References

[1] Seung-Hwan Baek and Felix Heide. Polka lines: Learning structured illumination and reconstruction for active stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5757–5767, 2021. 2

[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 4

[3] Basler dart daA1280-54uc. Basler AG, 2024. Rev. 85. 5

[4] Henryk Blasinski, Joyce Farrell, Trisha Lian, Zhenyi Liu, and Brian Wandell. Optimizing image acquisition systems for autonomous driving. Electronic Imaging, 2018(5):161– 1, 2018. 2

[5] Julie Chang and Gordon Wetzstein. Deep optics for monocular depth estimation and 3D object detection. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 10193–10202, 2019. 2

[6] Julie Chang, Vincent Sitzmann, Xiong Dun, Wolfgang Heidrich, and Gordon Wetzstein. Hybrid optical-electronic convolutional neural networks with optimized diffractive optics for image classification. Scientific Reports, 8(1):12324, 2018. 1, 2

[7] Juhyung Choi, Jinnyeong Kim, Seokjun Choi, Jinwoo Lee, Samuel Brucker, Mario Bijelic, Felix Heide, and Seung-Hwan Baek. Dual exposure stereo for extended dynamic range 3D imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6283–6293, 2025. 3, 12

[8] Geoffroi Cotˆ e, Fahim Mannan, Simon Thibault, Jean-´ Franc¸ois Lalonde, and Felix Heide. The differentiable lens: Compound lens search over glass surfaces and materials for object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20803–20812, 2023. 1, 2

[9] Steven Diamond, Vincent Sitzmann, Frank Julca-Aguilar, Stephen Boyd, Gordon Wetzstein, and Felix Heide. Dirty Pixels: Towards end-to-end image processing and perception. ACM Transactions on Graphics, 40(3):1–15, 2021. 2

[10] FLIR FLEA®3 USB3 Vision. FLIR Integrated Imaging Solutions Inc., 2017. Rev. 8.1. 8

[11] Alessandro Foi. Clipped noisy images: Heteroskedastic modeling and practical denoising. Signal Processing, 89 (12):2609–2629, 2009. 2

[12] Bhargav Ghanekar, Salman Siddique Khan, Pranav Sharma, Shreyas Singh, Vivek Boominathan, Kaushik Mitra, and Ashok Veeraraghavan. Passive snapshot coded aperture dualpixel RGB-D imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25348–25357, 2024. 1, 2, 6, 7, 14

[13] Harel Haim, Shay Elmalem, Raja Giryes, Alex M Bronstein, and Emanuel Marom. Depth estimation from a single image using deep learned phase coded mask. IEEE Transactions on Computational Imaging, 4(3):298–310, 2018. 2

[14] Lei He, Guanghui Wang, and Zhanyi Hu. Learning depth from single images with deep neural network embedding fo-

cal length. IEEE Transactions on Image Processing, 27(9): 4676–4689, 2018. 2

[15] Yunzhong Hou, Xingjian Leng, Tom Gedeon, and Liang Zheng. Optimizing camera configurations for multi-view pedestrian detection. arXiv preprint arXiv:2312.02144, 2023. 2

[16] Hayato Ikoma, Cindy M Nguyen, Christopher A Metzler, Yifan Peng, and Gordon Wetzstein. Depth from defocus with learned optics for imaging and occlusion-aware depth esti mation. In 2021 IEEE International Conference on Computational Photography, pages 1–12. IEEE, 2021. 1, 2, 3, 4, 5, 6, 7, 8, 11, 13

[17] Eric Jang, Shixiang Gu, and Ben Poole. Categorical repa rameterization with gumbel-softmax. In International Con ference on Learning Representations, 2017. 3

[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ George Drettakis, et al. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4):139–1, 2023. 1, 2

[19] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd Interna tional Conference on Learning Representations, 2015. 3, 5

[20] Tzofi Klinghoffer, Kushagra Tiwary, Nikhil Behari, Bhavya Agrawalla, and Ramesh Raskar. DISeR: Designing imaging systems with reinforcement learning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vi sion, pages 23632–23642, 2023. 1, 2

[21] Anat Levin, Rob Fergus, Fredo Durand, and William T Free-´ man. Image and depth from a conventional camera with a coded aperture. ACM Transactions on Graphics, 26(3):70– es, 2007. 2, 7, 8, 11

[22] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 4

[23] Zhenyi Liu, Trisha Lian, Joyce Farrell, and Brian Wan dell. Soft prototyping camera designs for car detection based on a convolutional neural network. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 0–0, 2019. 2

[24] Zhenyi Liu, Shen Minghao, Jiaqi Zhang, Shuangting Liu, Henryk Blasinski, Trisha Lian, and Brian Wandell. A system for generating complex physically accurate sensor images for automotive applications. Electronic Imaging, 2019: 53–1, 2019.

[25] Zhenyi Liu, Devesh Shah, Alireza Rahimpour, Devesh Upadhyay, Joyce Farrell, and Brian A Wandell. Using simulation to quantify the performance of automotive perception systems. arXiv preprint arXiv:2303.00983, 2023. 2

[26] Julien NP Martel, Lorenz K Mueller, Stephen J Carey, Piotr Dudek, and Gordon Wetzstein. Neural Sensors: Learning pixel exposures for hdr imaging and video compressive sensing with programmable sensors. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(7):1642–1653, 2020. 2

[27] Christopher A Metzler, Hayato Ikoma, Yifan Peng, and Gordon Wetzstein. Deep optics for single-shot high-dynamicrange imaging. In Proceedings ofthe IEEE/CVF Conference

on Computer Vision and Pattern Recognition, pages 1375– 1385, 2020. 2

[28] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local Light Field Fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics, 38(4):1–14, 2019. 4

[29] Henrique Morimitsu, Xiaobin Zhu, Roberto M Cesar, Xiangyang Ji, and Xu-Cheng Yin. RAPIDFlow: Recurrent adaptable pyramids with iterative decoding for efficient optical flow estimation. In 2024 IEEE International Conference on Robotics and Automation, pages 2946–2952. IEEE, 2024. 3, 11, 12

[30] Cindy M Nguyen, Julien NP Martel, and Gordon Wetzstein. Learning spatially varying pixel exposures for motion deblurring. In 2022 IEEE International Conference on Computational Photography, pages 1–11. IEEE, 2022. 2

[31] Nicolas Robidoux, Luis E Garcia Capel, Dong-eun Seo, Avinash Sharma, Federico Ariza, and Felix Heide. End-toend high dynamic range camera pipeline optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6297–6307, 2021. 2

[32] Prasan A Shedligeri, Sreyas Mohan, and Kaushik Mitra. Data driven coded aperture design for depth recovery. In 2017 IEEE International Conference on Image Processing, pages 56–60. IEEE, 2017. 2, 11

[33] Vincent Sitzmann, Steven Diamond, Yifan Peng, Xiong Dun, Stephen Boyd, Wolfgang Heidrich, Felix Heide, and Gordon Wetzstein. End-to-end optimization of optics and image processing for achromatic extended depth of field and superresolution imaging. ACM Transactions on Graphics, 37(4): 1–13, 2018. 1, 2

[34] Qilin Sun, Congli Wang, Fu Qiang, Dun Xiong, and Heidrich Wolfgang. End-to-end complex lens design with differentiable ray tracing. ACM Transactions on Graphics, 40 (4):1–13, 2021. 2

[35] Kushagra Tiwary, Aaron Young, Zaid Tasneem, Tzofi Klinghoffer, Akshat Dave, Tomaso Poggio, Dan-Eric Nilsson, Brian Cheung, and Ramesh Raskar. What if Eye...? Computationally recreating vision evolution. Science Advances, 11(51):eady2888, 2025. 1, 2

[36] Ethan Tseng, Felix Yu, Yuting Yang, Fahim Mannan, Karl ST Arnaud, Derek Nowrouzezahrai, Jean-Franc¸ois Lalonde, and Felix Heide. Hyperparameter optimization in black-box image processing using differentiable proxies. ACM Transactions on Graphics, 38(4):27–1, 2019. 2

[37] Ethan Tseng, Ali Mosleh, Fahim Mannan, Karl St-Arnaud, Avinash Sharma, Yifan Peng, Alexander Braun, Derek Nowrouzezahrai, Jean-Francois Lalonde, and Felix Heide. Differentiable compound optics and processing pipeline optimization for end-to-end camera design. ACM Transactions on Graphics, 40(2):1–19, 2021. 1, 2

[38] Ashok Veeraraghavan, Ramesh Raskar, Amit Agrawal, Ankit Mohan, and Jack Tumblin. Dappled Photography: Mask enhanced cameras for heterodyned light fields and coded aperture refocusing. ACM Transactions on Graphics, 26(3):69, 2007. 2

[39] Brian Wandell, David Cardinal, David Brainard, Zheng Lyu, Zhenyi Liu, Aaron Webster, Joyce Farrell, Trisha Lian, Hen ryk Blasinski, and Kaijun Feng. ISETCam, 2024. 2

[40] Brian Wandell, David Cardinal, Trisha Lian, Zhenyi Liu, Zheng Lyu, David Brainard, Henryk Blasinski, Amy Ni, Joelle Dowling, Max Furth, Thomas Goossens, Anqi Ji, and Jennifer Maxwell. ISET3D, 2024. 2

[41] Korbinian Weikl, Damien Schroeder, Daniel Blau, Zhenyi Liu, and Walter Stechele. End-to-end imaging system opti mization for computer vision in driving automation. In Proceedings of IS&T Int’l. Symp. on Electronic Imaging: Au tonomous Vehicles and Machines, 2021. 2

[42] Yicheng Wu, Vivek Boominathan, Huaijin Chen, Aswin Sankaranarayanan, and Ashok Veeraraghavan. Phase-Cam3D—Learning phase masks for passive single view depth estimation. In 2019 IEEE International Conference on Computational Photography, pages 1–12. IEEE, 2019. 2

[43] Chengyang Yan and Donald G. Dansereau. TaCOS: Task specific camera optimization with simulation. In Proceed ings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2052–2062, 2025. 1, 2, 4, 5, 6, 13

[44] Chengyang Yan, Mitch Bryson, and Donald G. Dansereau. JOCA: Task-driven joint optimisation of camera hardware and adaptive camera control algorithms. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3928–3938, 2026. 1, 2

[45] Xinge Yang, Qiang Fu, Yunfeng Nie, and Wolfgang Heidrich. Image quality is not all you want: Task-driven lens design for image classification. 2023. 1, 2

[46] Xinge Yang, Qiang Fu, and Wolfgang Heidrich. Curriculum learning for ab initio deep learned refractive optics. Nature Communications, 15(1):6572, 2024. 2

[47] Vickie Ye, Ruilong Li, 1 Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for Gaussian splatting. Journal of Ma chine Learning Research, 26(34):1–17, 2025. 4

[48] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018. 5

[49] Yuxuan Zhang, Bo Dong, and Felix Heide. All you need is raw: Defending against adversarial attacks with camera image pipelines. In European Conference on Computer Vision, pages 323–343. Springer, 2022. 2

[50] Changyin Zhou, Stephen Lin, and Shree Nayar. Coded aper ture pairs for depth from defocus. In 2009 IEEE 12th International Conference on Computer Vision, pages 325–332. IEEE, 2009. 2

![](images/878f30ad2b74fa9c54e940c8cb0fa1f0ec0b8bfc1027010cdaf442186a1ba4e2.jpg)  
Figure 6. The monocular depth-from-defocus method. The backbone follows the U-Net of Ikoma et al. [16], which takes a coded aperture image and its Tikhonov-deconvolved focal stack as input and jointly predicts the AiF image and depth map. We augment this backbone with a frequency injection module. In this module, the input image is divided into patches, a 2D Fourier transform is applied to each patch independently, and the log-magnitude spectrum is computed to capture local, spatially-varying defocus strength. The per-patch spectra are encoded by a lightweight 3-layer convolutional network, reassembled into a 2D feature map based on each patch’s location, and injected at the bottleneck of the U-Net.

## A. Depth-from-Defocus Networks

We provide additional details of the depth-from-defocus networks used as perception tasks in our experiments. The structures of both the monocular and stereo versions are described in this section.

## A.1. Monocular

As illustrated in Figure 6, our monocular task network builds upon the depth-from-defocus architecture of [16]. The original method first applies a Tikhonov inverse filter to the rendered coded aperture images, using the PSFs to deconvolve them into a focal stack; each plane of this stack contains a relatively sharp region corresponding to that plane’s PSF. However, because the forward model for defocus blur simulation is non-linear, the Tikhonov inverse cannot directly invert this process to obtain the AiF image. The focal stack, together with the rendered coded aperture image, is therefore used as input to a convolutional neural network with a U-Net architecture, which outputs both the AiF image and depth map. Details of this architecture can be found in [16].

We modify this model by incorporating a frequency injection module, as shown in Figure 6, that encodes spatiallylocal defocus cues directly from the coded image in the frequency domain for better depth separation as in [21]. Rather than transforming the entire image globally, we follow [32] and operate on local patches. The rendered blurry image b is partitioned into overlapping $P \times P$ windows with stride S via reflection-padded unfolding, yielding $N _ { h } \times N _ { w }$ patches $\{ { \bf b } _ { i j } \}$ . Each patch is independently transformed via the real-valued 2D Fourier transform and its log-magnitude spectrum log $\left( 1 + | \mathcal { F } \{ \mathbf { b } _ { i j } \} | \right)$ is computed to obtain local defocus cues. A lightweight 3-layer convolutional encoder with shared weights maps each patch spectrum to a descriptor via global average pooling; these are then reassembled into a dense feature map preserving the spatial correspondence between each descriptor and its originating image region. This map is then injected into the U-Net bottleneck via a learned per-channel gated addition:

$$
\mathbf { x }  \mathbf { x } + \sigma ( \mathbf { g } ) \odot \mathbf { f } _ { \mathrm { f r e q } } ,\tag{4}
$$

where g is initialised $\mathrm { t o } - 3$ so that the branch contributes negligibly at the start of training and the network learns the appropriate mixing ratio. Fusing at the bottleneck allows the spatially-structured frequency cues to condition the entire decoder without perturbing the fine-grained spatial skip connections.

## A.2. Stereo

For the stereo camera setup, we reuse the U-Net architecture with frequency injection from the monocular version as the base model, with modifications to its input. An illustration is displayed in Figure 7.

We first estimate the disparity between the two views with a pretrained optical flow network, RAPIDFlow [29], following [7]; the network remains frozen during training. To obtain reliable optical flow estimates, we first construct an intermediate all-in-focus image per view by fusing the Tikhonov-inverted focal stack via a differentiable Laplacian sharpness measure, which improves photometric consistency across the two views. The fused images are then passed to RAPIDFlow to predict a dense disparity map, which warps the right-view focal stack into the left view’s coordinates via bilinear grid sampling. This provides additional photometric and defocus cues for AiF reconstruction and depth estimation. The left focal stack, the warped right focal stack, and a depth map derived from the disparity estimate are used as inputs to the same U-Net architecture as in the monocular setting, with the input shape modified accordingly; the network outputs the reconstructed AiF image and depth map for the left view.

![](images/05958d0c19160f85508c3a355d2020f5d42663a54787b19d44d7fc2bda0d4a4c.jpg)  
Figure 7. The stereo depth-from-defocus method. Per-view AiF proxy images are synthesised by fusing the Tikhonov-deconvolved focal stack using a differentiable Laplacian sharpness measure and passed to a frozen RAPIDFlow [29] network to estimate dense disparity. The disparity is then used to warp the right-view into the left-view coordinate frame via bilinear grid sampling. The left-view focal stack, the warped right-view focal stack, and a depth map derived from the disparity estimate are concatenated and fed into the monocular U-Net backbone with frequency injection, yielding the reconstructed AiF image and depth map for the left view.

## B. Additional Qualitative Results on Monocular Camera Design

We provide further qualitative comparisons for the monocular camera design experiment under the lower-SNR (Noise×200) scenario in Fig. 8. These results are consistent with the Noise×100 scenario reported in the main paper: our method achieves better AiF reconstruction and depth estimation than the existing baselines. This further supports the effectiveness of jointly optimising camera hardware and perception under challenging noise conditions.

## C. Additional Qualitative Results on Stereo Camera Design

Fig. 9 presents full qualitative comparisons between our method and the baselines across both noise scenarios. Our stereo setup achieves higher AiF reconstruction quality by combining photometric cues from both views under strong defocus blur, with one view focused on the foreground and the other on the background. Its depth estimation is likewise improved by exploiting disparity as an additional cue. The baseline methods, in contrast, are limited to a single focal distance by their hardware, resulting in lower AiF reconstruction quality and depth estimation accuracy under strong defocus blur.

![](images/0b403865edac297b9b07640bef469f7b06ad7455c160b007b92a7fd87982312e.jpg)  
Figure 8. Qualitative comparison for the monocular experiment under the Noise×200 scenario. Our jointly optimised camera, mask, and perception model again produce sharp AiF images and accurate depth, while the fixed-camera baseline (Ikoma et al. [16]) and the derivative-free joint optimiser (TaCOS [43]) produce images with more defocus blur and lower depth accuracy in these regions.

## D. Additional Details on Real-World Experiments

We validate our method with a real-world prototype of a coded aperture camera. As discussed in the main paper, we use a commercial FLIR Flea3 camera with a Fujinon DF6HA-1B lens due to manufacturing constraints. The lens has a fixed focal length of 6.15 mm; we additionally maximise the entrance pupil diameter by setting the f-number to 1.2, which eases production of the coded aperture mask. We optimise the coded aperture mask and the lens’s focus distance for both the monocular and stereo experiments, with the mask set to a 13×13 grid. A relay lens (Edmund Optics #46-007) then images the manufactured mask onto the entrance pupil of the main lens, placing the coded aperture at the pupil plane. The setup is shown in Figure 5 of the main paper.

The manufactured mask’s PSF is measured by imaging a point light source at a known focus distance and source-tocamera distance. This measured PSF replaces its simulated counterpart in the image formation model, and we fine-tune the task network on the same training data for 1000 steps. Figure 10 (a) shows the optimised coded aperture mask for both monocular and stereo setups, the manufactured masks, and their measured PSFs.

Figure 10 (b) shows the captured coded aperture image, reconstructed AiF image, and estimated depth map for the monocular setup, while the corresponding stereo results appear in Figure 5 of the main paper. Our method reconstructs sharp AiF images and plausible depth maps under real optical conditions, though the depth estimation lags behind the simulated results, with visible errors near the scene’s midplane. We attribute this shortfall to manufacturing tolerances and the broader gap between simulated and captured imagery. Comparing Figure 10 (b) with Figure 5 of the main paper, the stereo configuration yields higher depth accuracy than the monocular one, in line with our simulated results.

![](images/189448d922b43e5ce1e4836b82424834ea863abc6128813d91a56d9745f0f4c2.jpg)  
Figure 9. Qualitative comparison of the proposed stereo method with a coded aperture dual-pixel camera method (CADS [12]) and the monocular alternative. Our method achieves better AiF reconstruction by combining varying focus planes across the two views, and better depth estimation by exploiting disparity as an additional cue.

(a)  
![](images/cb38fe47a523958762d98cc34baac3528eb2575e34faa9ff2ff5f95e3360252e.jpg)  
(b)  
Figure 10. (a) The optimised coded aperture masks in both monocular and stereo experiments, the manufcatured masks, and their measured PSFs. (b) The captured coded aperture image, reconstructed AiF image, and estimated depth map for the monocular setup. The AiF and depth recovery on real hardware show plausible outcomes but with modest accuracy gap attributable to the sim-to-real domain shift and manufacturing tolerances.