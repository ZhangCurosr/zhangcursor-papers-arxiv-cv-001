# PAGS: Autofocusing Photoacoustic Tomography via Speed-of-Sound-Adaptive Gaussian Splatting

Jiarui Ge<sup>1</sup> · Jintao Ma<sup>2</sup> · Bangxu Fan<sup>2</sup> · Jinyan Zhang<sup>2</sup> · Xiaokang Yang<sup>1</sup> · Shuai Na<sup>2,3,4,5,∗</sup> · Xiaoyun Yuan<sup>1,∗</sup>

Abstract Photoacoustic computed tomography (PACT) combines optical absorption contrast with acoustic detection for high-resolution deep-tissue imaging. A persistent challenge is that unknown speed-of-sound (SoS) heterogeneity changes acoustic time-of-flight, causing defocusing artifacts when reconstruction assumes a uniform SoS. Existing SoS-adaptive methods either rely on calibrated acoustic priors or optimize dense physical medium models, which becomes expensive and dificult to scale in 3D. We propose PAGS, a diferentiable framework for blind autofocusing PACT via speed-ofsound-adaptive Gaussian splatting. PAGS represents the initial pressure field with sparse Gaussian photoacoustic (PA) sources and replaces explicit medium recovery with a compact anisotropic path-averaged SoS (ASoS) field parameterized by spherical harmonic probes. This latent propagation field directly controls source-to-transducer arrival-time alignment, while an analytic Gaussian acoustic projection maps the source representation to transducer signals eficiently. The resulting closed-loop signaldomain optimization jointly updates the Gaussian PA source parameters and the ASoS field from measured data, without calibrated SoS priors. Experiments on simulated and physical phantom data demonstrate im-

proved reconstruction sharpness under heterogeneous acoustic media, robustness to sparse-view sampling, and computational benefits from the analytic Gaussian projection.

Keywords Photoacoustic Computed Tomography Gaussian Splatting · Speed-of-Sound Adaptation · Autofocus

Electronic Supplementary Material Supplementary Video 1 provides rotating three-dimensional comparisons of vanilla UBP, Dual-SoS UBP, SlingBAG, and PAGS, together with PAGS reconstructions at diferent sampling ratios.

## 1 Introduction

Photoacoustic computed tomography (PACT) is a noninvasive imaging modality that combines optical absorption contrast with acoustic detection [1]. As illustrated in Fig. 1, a short laser pulse excites light-absorbing chromophores, such as hemoglobin or melanin, and the resulting thermoelastic expansion generates broadband ultrasonic waves that are recorded by surrounding transducers. Reconstructing the initial pressure field from these time-resolved acoustic measurements enables highresolution structural and functional mapping of biological tissues. This capability has made PACT useful in a wide range of preclinical and clinical studies, including vascular imaging, neuroimaging, and tumor characterization.

The quality of PACT reconstruction, however, is strongly afected by the assumed speed of sound (SoS). Standard analytical reconstruction methods, such as universal back-projection (UBP) [2], commonly assume a globally uniform SoS. Real tissue and surrounding coupling media often have diferent acoustic speeds, so the same photoacoustic source can have diferent pathaveraged SoS values toward diferent transducers [1, 3]. These path-dependent arrival-time errors lead to defocusing, geometric distortion, and loss of fine vascular detail. Correcting unknown SoS-induced aberration is therefore a central problem for high-resolution PACT.

![](images/b4eb6b1407192eb3671d5284f53e0876bb7957417a8d5beb84a26cf857a1205e.jpg)  
Fig. 1 Overview of the proposed PAGS framework for blind autofocusing photoacoustic tomography. Compared with universal back-projection (UBP) under a uniform speed-of-sound (SoS) assumption, PAGS represents the initial pressure field with sparse Gaussian photoacoustic (PA) sources and learns an anisotropic path-averaged SoS (ASoS) field with spherical harmonic (SH) probes, enabling focused reconstruction without calibrated SoS priors.

Existing SoS-adaptive reconstruction strategies face a dificult trade-of. Prior-informed multi-SoS reconstruction can improve focusing when tissue boundaries and regional SoS values are known, but such calibrated acoustic priors are often unavailable and may require additional segmentation or auxiliary measurements. More general iterative or learning-based approaches can optimize a dense physical medium or scalar SoS map jointly with the absorption distribution [3, 4], but the arrivaltime correction needed for focusing is obtained only after path integration through that medium. This highdimensional parameterization and indirect forward dependency make blind 3D optimization computationally expensive and prone to poor local solutions.

We propose PAGS, a diferentiable framework for blind autofocusing PACT via speed-of-sound-adaptive Gaussian splatting. Building on Gaussian-based reconstruction [5,6], the key idea is to replace explicit recovery of the full acoustic medium with a latent anisotropic path-averaged SoS (ASoS) field that directly controls source-to-transducer arrival-time alignment. PAGS represents the initial pressure field with sparse Gaussian photoacoustic (PA) sources and represents the propagation efect with compact spherical harmonic (SH) probes anchored on a spatial grid. For each Gaussian PA source and transducer direction, PAGS queries the ASoS field, projects the source to the transducer signal using an analytic Gaussian acoustic model, and optimizes the source and propagation parameters in a single signal-domain loop.

This design aligns the representation with the structure of the inverse problem. The PA source distribution is sparse and localized, making Gaussian PA sources a natural continuous primitive for vascular structures. The heterogeneous acoustic medium is not recovered directly; instead, its accumulated efect on propagation is expressed as a smoother, lower-dimensional ASoS field over source locations and transducer directions. As a result, PAGS focuses on the propagation variable that directly afects reconstruction sharpness while avoiding the cost and ambiguity of dense medium recovery.

The main contributions are summarized as follows:

1. We introduce a latent ASoS propagation field for blind PACT autofocusing. It models direction-varying path-averaged SoS directly, avoiding explicit dense acoustic medium recovery and calibrated SoS priors.

2. We couple the ASoS field with sparse Gaussian PA sources and an analytic Gaussian acoustic projection, enabling eficient diferentiable optimization from measured transducer signals.

3. We validate PAGS on simulated and physical phantom data, including sparse-view settings and component ablations, showing improved focusing under heterogeneous acoustic media and favorable eficiency from the analytic projection.

## 2 Related Work

## 2.1 PACT Reconstruction Methods

Conventional PACT reconstruction algorithms are broadly categorized into analytical and iterative paradigms. Analytical methods, predominantly Universal Back-Projection (UBP) [2] and Delay-and-Sum (DAS), are widely adopted for their computational eficiency. By deterministically mapping time-resolved signals to the spatial domain, they operate on the highly idealized assumption of straight-ray propagation within a homogeneous medium. However, this assumption critically fails in realistic biological tissues [1, 3]. Acoustic heterogeneities across diverse tissue layers induce wavefront refraction and significant time-of-flight (ToF) mismatches. Since manual global SoS tuning cannot compensate for these localized phase errors, analytical reconstructions inherently sufer from geometric distortions, spatial blurring, and severe resolution degradation [7]. Furthermore, these methods strictly rely on dense spatial sampling to guarantee reconstruction fidelity. Under sparse sensor coverage or limited-view geometries, they lack the inherent mechanisms to constrain the solution space, inevitably generating streak artifacts and making reliable structure identification exceedingly dificult [1, 6].

To address these limitations, model-based iterative reconstruction methods treat PACT as an inverse problem under an optimization framework [8–10]. By integrating regularization terms such as Total Variation (TV) or L -norm penalties, these frameworks achieve superior resilience against sparse sampling and measurement noise. For example, Huang et al. [8] developed a full-wave iterative reconstruction framework that explicitly accounts for acoustic heterogeneity, demonstrating that incorporating true SoS maps significantly improves image fidelity in realistic tissue models. More recent works further introduce learned regularization priors [9] and multi-channel autoencoder constraints [10] to accel erate convergence and enhance robustness. Time-reversal approaches [11] also exploit the symmetry of the wave equation for eficient reconstruction, though they are still sensitive to SoS mismatches. However, the repeated evaluation of the forward acoustic propagation and its adjoint gradient inherently incurs substantial computational and memory costs. Furthermore, standard iterative frameworks typically rely on a predefined SoS map for accurate forward modeling. In realistic biological tissues, acquiring the exact anatomical SoS distribution is highly challenging. Inaccuracies in this static prior can compromise the compensation of acoustic aberrations, ultimately limiting the overall reconstruction fidelity [12].

In recent years, deep learning has emerged as a powerful alternative for PACT reconstruction. Data-driven architectures, ranging from U-Nets to difusion models, leverage rich learned priors to achieve striking artifact reduction under ill-posed conditions, such as sparse-view or limited-aperture scenarios [1, 12–14]. While these purely data-driven networks ofer impressive structural restoration, pioneering physics-aware frameworks have further attempted to explicitly incorporate physical constraints to dynamically resolve unknown acoustic aberrations. For instance, Hauptmann et al. [15] proposed a model-based learning scheme that couples an iterative inversion with a learned regularization, achieving high-quality limited-view 3D reconstructions. Poimala et al. designed a specialized network to compensate for unknown SoS distributions, efectively recovering spatial resolution degraded by wavefront aberrations [3]. Similarly, Li et al. employed coordinate-based implicit neural representations to directly recover a spatial SoS field jointly with the absorption distribution, enabling aberration correction through physically informed decoding [4]. Together, these learning-based and physics-aware approaches demonstrate the immense potential of integrating data-driven priors with acoustic physical models. However, despite their success in accounting for non-uniform SoS, representing the acoustic field via dense grids or implicit functions remains computationally demanding, limiting their scalability and memory eficiency in complex 3D environments.

## 2.2 Gaussian-based Reconstruction Methods

3D Gaussian Splatting (3DGS) [5, 16–18] has emerged as a powerful explicit scene representation paradigm. Instead of relying on traditional voxels or meshes, it parameterizes a scene using a collection of learnable, anisotropic 3D Gaussian primitives. Each primitive is explicitly defined by its position, covariance, opacity, and view-dependent color. Rendered via a highly eficient diferentiable tile-based rasterizer, 3DGS achieves real-time performance while preserving state-of-the-art visual fidelity. Recent extensions have further improved anti-aliasing [16], geometric accuracy [17], and structured view-adaptive rendering [18], solidifying 3DGS as a robust spatial modeling framework. This structural flexibility and computational eficiency have inspired a surge of dynamic extensions. For instance, Deformable 3DGS [19] introduces a per-Gaussian temporal deformation field, allowing the framework to reconstruct high-quality geometry and appearance from monocular dynamic videos. Building upon the concept of temporal modeling, 4DGS [20] directly embeds the optimization into the space-time domain, jointly updating Gaussian attributes and motion trajectories to enable real-time dynamic view synthesis. Furthermore, Spacetime Gaussian Feature Splatting [21] explicitly models the evolution of both geometry and appearance using polynomial trajectories, which significantly enhances the temporal coherence of dynamic renderings. Collectively, these advances highlight the core strengths of Gaussian-based representations. As explicit, grid-free primitives, they can eficiently capture fine geometric details, seamlessly adapt to complex topological changes, and maintain full end-to-end diferentiability, establishing 3DGS as a highly expressive and robust spatial modeling framework.

Building upon this momentum, Gaussian representations have been rapidly adapted to medical tomographic reconstruction. In X-ray computed tomography (CT), R<sup>2</sup>-Gaussian [22] introduced a radiometrically corrected Gaussian splatting forward model that compensates for scattering and beam hardening efects, achieving high-quality 3D reconstructions under sparse-view acquisition. Similarly, for 3D ultrasound volume imaging, UltraGauss [23] integrated learnable Gaussian primitives with an acoustic wave propagation renderer to build a fast reconstruction framework, demonstrating strong adaptability to reflection and attenuation physics. In whole-body positron emission tomography (PET), GR-Difusion [24] combines Gaussian representations with difusion models to improve low-dose reconstruction. MedGS [25] extends Gaussian splatting to unified multi-modal 3D medical imaging. These successful crossmodal adaptations confirm that, when equipped with modality-specific diferentiable forward models, Gaussian primitives can serve as a highly expressive and versatile basis for solving complex volumetric inverse problems.

In the realm of PACT, SlingBAG [6] recently pioneered the integration of Gaussian representations by parameterizing the initial photoacoustic pressure as a collection of learnable 3D Gaussian PA sources. For acoustic forward projection, it approximates each continuous Gaussian primitive by multiple concentric spherical shells and uses a smoothed sphere response with error-function evaluations to maintain diferentia bility. While efective, this approximation increases the per-iteration computational cost and still fundamentally hinges on a globally homogeneous SoS assumption. Under this constraint, the acoustic time-of-flight from each Gaussian PA source to each detector is evaluated solely from Euclidean distance, neglecting path-dependent aberrations caused by heterogeneous acoustic media. Consequently, the uniform-SoS limitation leads to structural defocusing and geometric distortion, compromising spatial resolution and reconstruction fidelity.

These limitations motivate PAGS, an SoS-adaptive Gaussian splatting framework for blind autofocusing PACT. PAGS keeps the sparse Gaussian PA source representation, replaces the multi-sphere approximation with an analytic Gaussian acoustic projection, and introduces a learnable anisotropic path-averaged SoS field parameterized by compact spherical harmonic (SH) probes. In this way, PAGS optimizes the source representation and the path-level propagation correction directly from measured transducer signals, without recovering a dense physical SoS map.

## 3 Method

## 3.1 Preliminaries

PAGS builds on the sparse source representation introduced by SlingBAG [6]. Given time-resolved acoustic measurements recorded by an array of transducers, the reconstruction target is the initial pressure field $p _ { 0 } ( \mathbf { r } )$ where r is a 3D coordinate in the imaging volume. Following SlingBAG, we represent this field as a set of Gaussian photoacoustic (PA) sources:

$$
p _ { 0 } ( \mathbf { r } ) = \sum _ { i = 1 } ^ { N } a _ { i } \exp \left( - \frac { \| \mathbf { r } - \mathbf { x } _ { i } \| ^ { 2 } } { 2 \sigma _ { i } ^ { 2 } } \right) ,\tag{1}
$$

where N is the number of Gaussian PA sources, and $\mathbf { x } _ { i } ,$ ${ { a } _ { i } } ,$ , and $\sigma _ { i }$ denote the center, pressure amplitude, and spatial scale of the i-th source. This representation is compact, continuous, and diferentiable, making it well suited for localized PA structures such as vessels.

However, directly reusing this Gaussian PA source backbone leaves heterogeneous acoustic propagation unaddressed. SlingBAG projects Gaussian PA sources to transducer signals under a globally uniform speed of sound (SoS), so its predicted time-of-flight depends only on source–transducer distance. When the actual propagation is heterogeneous, this uniform-SoS assumption introduces arrival-time errors and leads to out-of-focus reconstruction. PAGS keeps the compact Gaussian PA source representation, but redesigns the propagation and forward-projection modules described below.

![](images/fd66b6bb2ab48ac48b8c382ce4dd6795858c164e3124157321827a394d945ab1.jpg)  
Fig. 2 Core implementation details of PAGS. (a) The diferentiable reconstruction model couples Gaussian photoacoustic (PA) sources, anisotropic path-averaged SoS coeficients, and transducer measurements through an analytic acoustic forward model. (b) The anisotropic path-averaged SoS (ASoS) field represents the direction-varying path-averaged SoS from each source location to the transducer array, implicitly capturing the accumulated propagation efect of heterogeneous media. (c) The ASoS query uses two interpolations: spherical harmonic (SH) coeficients are first obtained at the PA source position by spatial interpolation from neighboring probes, and the path-averaged SoS is then obtained by angular SH interpolation along each source–transducer direction.

## 3.2 Framework Overview

PAGS formulates blind autofocusing as a closed-loop diferentiable signal-matching problem. As illustrated in Fig. 2(a), its learnable variables are the Gaussian PA source parameters and the propagation parameters that map these sources to the measured transducer signals. A forward pass starts from the current Gaussian PA sources, queries the anisotropic path-averaged SoS (ASoS) field for each source–transducer pair, projects each source with an analytic acoustic model, and super poses all source contributions to synthesize the detector signals.

The signal residual then updates both parts of the representation. The Gaussian PA source parameters refine the reconstructed initial pressure field, while the ASoS field corrects the arrival-time alignment that controls focusing. Thus, PAGS difers from open-loop reconstruction by allowing source reconstruction and propagation correction to refine each other within the same signal-domain objective.

The central contribution of PAGS is to replace explicit acoustic medium recovery with a latent ASoS propagation field for autofocusing. Existing SoS-adaptive methods often optimize a dense medium or scalar SoS map, and the arrival-time correction needed for reconstruction is obtained only after path integration through that map. This creates a high-dimensional parameterization with an indirect forward dependency: a local change in the medium afects image focus only after being accumulated along many source–transducer paths. PAGS instead learns the anisotropic path-averaged SoS between source locations and transducer directions. Each ASoS value already represents the accumulated propagation efect along one path, yielding a smoother and lower-dimensional optimization target whose value directly changes the predicted arrival time. Coupled with the analytic Gaussian acoustic projection, this formulation lets signal residuals jointly update the Gaussian PA source parameters and the ASoS field through a simpler diferentiable reconstruction model.

The following subsections describe the main components in order: Sec. 3.3 defines the ASoS field and its SH probe parameterization, Sec. 3.4 derives the analytic acoustic projection of Gaussian PA sources, and Sec. 3.5 presents the joint optimization objective.

## 3.3 Anisotropic Path-Averaged SoS Field

We now define the ASoS field used by the propagation component. The ASoS field is a latent propagation representation whose queried value serves as the efective path-averaged SoS for converting a source–transducer distance into its acoustic delay. It is optimized to compensate for propagation-induced arrival-time errors in autofocusing, rather than to recover the local physical SoS distribution of the acoustic medium.

For a Gaussian PA source at x and a transducer at z, let

$$
R = \| { \bf z - x } \| , \qquad \omega = \frac { { \bf z - x } } { R }\tag{2}
$$

denote the source–transducer distance and propagation direction. The corresponding time-of-flight (ToF) is modeled as

$$
\mathrm { T o F } ( { \bf x } , { \bf z } ) = \frac { R } { v _ { \mathrm { a v g } } ( { \bf x } , \omega ) } ,\tag{3}
$$

where $v _ { \mathrm { a v g } } ( \mathbf { x } , \omega )$ is the efective path-averaged SoS from source position x along direction ω.

Here, “anisotropic” describes direction variation in the path-averaged SoS observed from a source location. As illustrated in Fig. 2(b), rays from the same source can cross diferent proportions of water and tissue before reaching diferent transducers; their path-averaged SoS values therefore vary with direction.

To make this continuous field compact and optimizable, PAGS stores it on a regular 3D grid of SoS SH probes, rather than attaching SoS parameters to individual Gaussian PA sources. This decouples the propagation model from the source set: Gaussian PA sources can move, split, and be pruned during optimization, while the ASoS field remains anchored in space and varies smoothly. Each probe represents the angular profile of the path-averaged SoS using second-order spherical harmonics (SH). We collect the nine real SH basis functions evaluated at direction ω as

$$
\begin{array} { r } { \phi _ { 2 } ( \omega ) = \left[ \phi _ { 0 } ( \omega ) , \ldots , \phi _ { 8 } ( \omega ) \right] ^ { \top } \in \mathbb { R } ^ { 9 } . } \end{array}\tag{4}
$$

The first basis function is a constant term, the next three basis functions capture first-order directional variation, and the remaining five basis functions capture secondorder angular variation. An SoS SH probe stores the corresponding weights $\mathbf { c } = [ c _ { 0 } , \hdots , c _ { 8 } ] ^ { \top } \in \mathbb { R } ^ { 9 } ;$ evaluating the weighted sum of these bases gives the path-averaged SoS for a queried direction.

At projection time, PAGS queries the ASoS value for each active Gaussian PA source and transducer direction. As shown in Fig. 2(c), the query has two steps. First, PAGS gathers the eight neighboring probes of the enclosing voxel and performs trilinear interpolation in weight space:

$$
\mathbf { c } ( \mathbf { x } ) = \sum _ { i , j , k \in \{ 0 , 1 \} } \mathbf { c } _ { i , j , k } \alpha _ { i , j , k } ( \mathbf { x } ) ,\tag{5}
$$

where $\alpha _ { i , j , k } ( { \bf x } )$ are the standard trilinear interpolation weights. This gives a local SH weight vector at the Gaussian PA source position. Second, PAGS evaluates the local SH expansion along the source–transducer direction:

$$
v _ { \mathrm { a v g } } ( \mathbf { x } , \omega ) = \mathbf { c } ( \mathbf { x } ) ^ { \top } \phi _ { 2 } ( \omega ) .\tag{6}
$$

Equations (5) and (6) therefore implement two interpolations: the former spatially interpolates the SH weights to the Gaussian PA source position, and the latter angularly evaluates the weighted SH bases along a transducer direction. During optimization, the queried ASoS is constrained to a fixed physically valid interval [v<sub>min</sub>, v<sub>max</sub>] to prevent nonphysical acoustic delays.

An additional benefit of this representation is memory eficiency. Consider a representative setting with 100K active Gaussian PA sources and 4.6K transducers. (1) Learnable parameters. A direct source-totransducer ASoS table would cost 100K×4.6K = 460M learnable scalars. PAGS instead optimizes the probe grid: $1 6 ^ { 3 }$ probes with 9 SH weights per probe require only 36.9K learnable weights, a 12.5K× reduction. (2) Query cache. If angular evaluation were performed first, the query would immediately materialize one ASoS scalar per source-to-transducer pair, costing another 460M intermediate values. PAGS performs spatial interpolation first, producing only one temporary 9D weight vector per active source, i.e., 100K×9 = 900K values, and evaluates the final source-to-transducer ASoS values on demand during signal synthesis. This reduces the intermediate query cache by about 510×, from 460M to 900K values. The queried $v _ { \mathrm { a v g } }$ is then passed to the analytic acoustic forward model described next.

## 3.4 Analytic Acoustic Forward Model

Given the path-averaged SoS queried from the ASoS field, PAGS next maps each Gaussian PA source to the transducer signal domain. SlingBAG maintains diferentiability by approximating a Gaussian PA source with multiple smoothed spherical shells, but this approximation introduces repeated error-function evaluations and increases per-iteration cost [6]. PAGS instead uses the continuous Gaussian PA source directly and evaluates its acoustic response in closed form.

For a single Gaussian PA source with amplitude a and spatial scale $\sigma ,$ the closed-form solution in an ideal uniform medium is

$$
\begin{array} { c } { \displaystyle p ( R , t ) = \frac { a } { 2 R } ( R - v _ { s } t ) \exp \left( - \frac { ( R - v _ { s } t ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } \\ { \displaystyle + \frac { a } { 2 R } ( R + v _ { s } t ) \exp \left( - \frac { ( R + v _ { s } t ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , } \end{array}\tag{7}
$$

where $v _ { s }$ is the constant SoS and R is the observation distance.

The first term is the outgoing wave and dominates at the detector. The second term corresponds to an incoming component whose envelope is $\exp ( - 2 R ^ { 2 } / \sigma ^ { 2 } )$ near the detection time $t \approx R / v _ { s }$ ; it is negligible in the far-field regime of PACT, where $R \gg \sigma$ . Under the straight-path ASoS approximation, we therefore keep the outgoing term and replace the constant $v _ { s }$ with the queried path-averaged SoS:

$$
p ( R , t ; v _ { \mathrm { a v g } } ) \approx \frac { a ( R - v _ { \mathrm { a v g } } t ) } { 2 R } \exp { \left( - \frac { ( R - v _ { \mathrm { a v g } } t ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } .\tag{8}
$$

This substitution treats the queried ASoS as an efective path-averaged SoS for the source–transducer path, so the analytic signal receives the learned arrival-time correction while retaining a closed-form diferentiable response.

Compared with the multi-sphere approximation, this expression uses a single exponential evaluation for each source, transducer, and time sample. It also remains diferentiable with respect to source position, amplitude, scale, and the ASoS coeficients through R and $v _ { \mathrm { a v g } } .$ Thus, each Gaussian PA source serves as both a compact image-domain primitive and an eficient unit for signaldomain optimization.

## 3.5 Joint Optimization

We now assemble the ASoS query and analytic acoustic projection into the signal-domain objective. For the ith Gaussian PA source and the m-th transducer, let $R _ { i , m } = \| \mathbf { z } _ { m } - \mathbf { x } _ { i } \|$ and

$$
\bar { v } _ { i , m } = v _ { \mathrm { a v g } } \left( \mathbf { x } _ { i } , \frac { \mathbf { z } _ { m } - \mathbf { x } _ { i } } { R _ { i , m } } \right) .\tag{9}
$$

Using $\bar { v } _ { i , m }$ in the source-adaptive response, the synthetic pressure signal at transducer $\mathbf { z } _ { m }$ is

$$
\hat { p } ( \mathbf { z } _ { m } , t ) = \sum _ { i = 1 } ^ { N } \frac { a _ { i } \varDelta _ { i , m } ( t ) } { 2 R _ { i , m } } \exp \left( - \frac { \varDelta _ { i , m } ^ { 2 } ( t ) } { 2 \sigma _ { i } ^ { 2 } } \right) ,\tag{10}
$$

where $\varDelta _ { i , m } ( t ) = R _ { i , m } - \bar { v } _ { i , m } t .$

Let $p _ { \mathrm { m e a s } } ( \mathbf { z } _ { m } , t )$ denote the measured pressure signal at the m-th transducer and time t. PAGS optimizes the

Gaussian PA source parameters and the ASoS probe coeficients by minimizing the signal-domain discrepancy:

$$
\mathcal { L } = \frac { 1 } { M T } \sum _ { m = 1 } ^ { M } \sum _ { t = 1 } ^ { T } \left( \hat { p } ( \mathbf { z } _ { m } , t ) - p _ { \mathrm { m e a s } } ( \mathbf { z } _ { m } , t ) \right) ^ { 2 } ,\tag{11}
$$

where M is the number of transducers and $T$ is the number of temporal samples.

Rather than introducing an explicit geometry-specific regularizer, PAGS controls the complexity of the ASoS field through its parameterization. Trilinear interpolation makes the queried field spatially continuous between probes, while the second-order SH basis limits angular variation to low-order modes. Thus, L is the complete optimization objective, with no assumption of a tworegion medium or prescribed tissue–water interface.

Optimization uses Adam over the full computation graph. The learnable variables include the Gaussian PA source parameters $\{ \mathbf { x } _ { i } , a _ { i } , \sigma _ { i } \}$ and the SH probe coefficients c of the ASoS field. For source adaptation, we follow the density-control strategy of SlingBAG, splitting high-gradient sources and pruning negligible ones so that the sparse source set evolves with the reconstructed structure. A bandpass filter can optionally be applied to measured signals before optimization when out-of-band system noise is significant.

## 4 Experiments

## 4.1 Experimental Setup

## 4.1.1 Datasets

We evaluate PAGS on two complementary datasets. The simulated dataset provides controlled geometry and a digital vascular phantom, while quantitative evaluation uses a reference reconstruction that incorporates realistic opto-acoustic and transducer responses. The physical phantom dataset tests robustness under real acquisition imperfections.

For the in silico dataset, we simulate acoustic measurements from a 3D vascular phantom using the k-space pseudospectral method implemented in k-Wave [26]. Fig. 3 shows the simulation configuration, reconstruction results, and corresponding digital phantom. The left panels display transducer positions as purple-to-yellow gradient dots (top) and the acquisition geometry with a grey spherical cap, a yellow ellipsoid SoS interface, and red phantom voxels of 0.1 mm edge length (bottom). The remaining columns compare three orthogonal views of the reconstructions and digital phantom, with a scale bar of 2 mm. The heterogeneous SoS map follows a dual-region setting with a 1450 m/s background and a 1550 m/s ellipsoidal inclusion centered with semi-axes of $7 , 8 ,$ and 11 mm. The detector array contains 4600 point transducers uniformly distributed on a spherical cap of radius 12.5 mm, with each detector recording 1024 samples at 50 MHz. The ground-truth initial pressure volume is generated at a voxel size of 0.1 mm and serves as the source for all simulations; it is not used directly as the quantitative evaluation target.

For the physical phantom dataset, the phantom contains absorptive structures and a tissue region with approximately 5%–8% acoustic contrast to the water. Measurements are acquired using a hemispherical array comprising 575 transducer elements. Data are collected at eight rotational views, yielding 4600 efective transducer positions in total. Each channel records 4096 samples at 20 MHz.

## 4.1.2 Baselines

We compare PAGS with three baselines that isolate diferent assumptions about PA source representation and SoS knowledge. All methods share identical data preprocessing steps.

Vanilla UBP. This baseline applies standard universal back-projection with a globally uniform SoS of 1450 $\mathrm { m } / \mathrm { s }$ Dual-SoS UBP. This oracle variant assumes a known tissue–water boundary and performs back-projection with the corresponding path-averaged SoS, without modeling refraction. It serves as a prior-informed analytical reference for SoS compensation.

Vanilla SlingBAG. This baseline follows the original SlingBAG algorithm [6], using 10 spheres to approximate each Gaussian PA source and assuming a globally uniform SoS during forward projection.

## 4.1.3 Evaluation

For simulated data, we report peak signal-to-noise ratio (PSNR), root mean square error (RMSE), and structural similarity index measure (SSIM). Direct voxel-wise comparison against the original binary phantom volume is confounded by the absence of realistic opto-acoustic and transducer responses and by the binary, high-contrast nature of the phantom (values restricted to 0 and 1). To better isolate the ability to correct SoS heterogeneity, we construct a reference volume by simulating acoustic propagation from the same vascular phantom under a uniform 1450 $\mathrm { m } / \mathrm { s }$ SoS using k-Wave, reconstructing with standard UBP, and peak-normalizing the result. All reconstructed volumes are peak-normalized in the same way. Because SoS mismatch can introduce a global geometric shift, we register each reconstruction to the reference via a rigid transformation (rotation and translation) estimated by maximizing normalized cross-correlation. This alignment corrects bulk positional errors without distorting local structures. The three metrics are then computed on these peak-normalized, rigidly registered volumes over a central region of interest spanning $x \in [ - 2 . 5 , 5 ]$ mm, $y \in [ - 5 , 5 ]$ mm, $z \in [ - 1 0 , 5 ]$ mm with an isotropic voxel size of 0.1 mm. For physical phantom data, where such a reference is unavailable, we evaluate reconstruction quality by visual inspection, local contrast, and line-profile widths of fine structures.

## 4.1.4 Implementation

PAGS is implemented in $\mathrm { P y }$ Torch. All experiments run on a graphics processing unit (GPU) workstation equipped with a single NVIDIA A100 (80 GB). The SH probe grid resolution is set between $8 ^ { 3 }$ and $1 6 ^ { 3 }$ depending on the physical size of the imaging scene, with the SH degree fixed at $D = 2 { \mathrm { ~ ( 9 ~ } }$ SH coeficients). Gaussian PA sources are initialized as approximately $1 0 ^ { 5 } – 1 0 ^ { 6 }$ primitives with random initial center positions. Optimization uses the Adam optimizer $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ ; the learning rate for Gaussian positions is $1 0 ^ { - 4 }$ , for amplitudes and scales around $1 0 ^ { - 3 }$ , and for SH coeficients $1 0 ^ { - 2 }$ . Training proceeds for 200–500 iterations. Adaptive density control is performed every 50 iterations, splitting highly contributing primitives and pruning those with negligible contribution based on their signal reconstruction impact.

## 4.2 In Silico Evaluation

We first evaluate PAGS on the in silico dataset. This controlled setting tests whether PAGS can correct reconstruction errors induced by a heterogeneous acoustic medium without using the tissue–water boundary as a calibrated prior. Fig. 3 compares the reconstructed vascular structures. Vanilla UBP sufers from visible geometric shift and defocus because a single uniform SoS cannot align the arrival times from diferent propagation paths. Vanilla SlingBAG benefits from a sparse Gaussian PA source representation and produces a cleaner background, but its uniform-SoS forward model still leaves noticeable blur and structural distortion. Dual-SoS UBP reduces part of the arrival-time mismatch by using the known tissue–water boundary, but it has two limitations. First, for eficient delay computation, the acoustic boundary must be represented by a simple geometry with tractable path-length calculation, such as an ellipsoid, which limits its ability to model more complex heterogeneous media. Second, it is an openloop analytical reconstruction: once the path-averaged delays are computed, the fixed back-projection operator cannot adapt to residual signal mismatch.

![](images/d0b5b91e527805c0dee4040e8a22e071386019e2ea41b66c22b8be89ffaf3154.jpg)  
Fig. 3 In silico configuration and results. Left: transducer positions (top) and acquisition geometry with a grey spherical cap, a yellow ellipsoid speed-of-sound interface, and red phantom voxels of 0.1 mm edge length (bottom). Middle: qualitative reconstruction comparison in three orthogonal views. Right: corresponding digital phantom. Scale bar: 2 mm. PAGS corrects artifacts induced by a heterogeneous acoustic medium without calibrated SoS priors, yielding sharper vascular structures than uniform-SoS baselines.

In contrast, PAGS reconstructs sharper and more continuous vessel branches while maintaining a clean background. This improvement comes from optimizing the Gaussian PA sources jointly with the ASoS field, so the signal residual can update both the Gaussian PA source parameters and the path-level propagation correction.

Table 1 reports PSNR, RMSE, and SSIM with respect to the uniform-SoS UBP reference. PAGS achieves the best scores among all compared methods, improving PSNR by 1.3 dB, reducing RMSE from 0.0354 to 0.0305, and increasing SSIM by 0.206 over vanilla SlingBAG. The substantial SSIM gain (from 0.331 to 0.537) indicates improved structural agreement rather than merely a visual sharpening efect.

Table 1 Quantitative evaluation on the in silico dataset, calculated against a uniform-SoS UBP reference after rigid registration and peak-intensity normalization.
<table><tr><td>Method</td><td>PSNR ↑</td><td>RMSE↓</td><td>SSIM ↑</td></tr><tr><td>Vanilla UBP</td><td>23.2</td><td>0.0695</td><td>0.230</td></tr><tr><td>Dual-SoS UBP (Oracle)</td><td>28.2</td><td>0.0388</td><td>0.384</td></tr><tr><td>Vanilla SlingBAG</td><td>29.0</td><td>0.0354</td><td>0.331</td></tr><tr><td>PAGS (Ours)</td><td>30.3</td><td>0.0305</td><td>0.537</td></tr></table>

## 4.3 Experimental Validation on a Physical Phantom

The in silico experiment validates PAGS under known geometry and a controlled reference reconstruction. We next evaluate whether the same blind autofocusing mechanism remains efective on physical phantom measurements, where the recorded signals include real system efects such as finite transducer bandwidth, electronic noise, acoustic attenuation, and element-dependent sensitivity. Since voxel-level ground truth is unavailable, Fig. 4 focuses on visual structure and local profile sharpness.

The global views in Fig. 4 show that vanilla UBP sufers from broad structural blur and reduced contrast, consistent with uncorrected SoS mismatch. Dual-SoS UBP improves part of the alignment by using a prescribed tissue–water boundary, but its simple boundary model and open-loop back-projection still leave residual defocus. Vanilla SlingBAG suppresses part of the difuse background through its sparse Gaussian PA source representation, yet its uniform-SoS forward model cannot fully focus the absorbers.

The zoomed regions and line-profile comparisons provide a more local view of this efect. The higher-contrast absorber responses indicate improved focusing rather than only a change in global intensity scaling. Neighboring structures that appear blurred or partially merged in the uniform-SoS baselines become more separable after joint source and ASoS optimization.

Taken together, the image-domain sharpening and improved local contrast suggest that PAGS can compensate for dominant arrival-time errors in real measurements while remaining a blind reconstruction method.

![](images/36e7d8ec082abb6191c968a854115d3d3eba47916ca54168516ec58654ccb33c.jpg)  
Fig. 4 Physical phantom reconstruction results, scale bar: 5 mm. The left panels compare global views, zoomed regions marked by red boxes, and local line-profile locations marked in yellow. The right panels show the normalized PA amplitude profiles along the marked lines. Compared with uniform-SoS UBP, prior-informed Dual-SoS UBP, and vanilla SlingBAG, PAGS produces sharper absorber boundaries, improved local contrast, and clearer separation between neighboring structures under blind SoS adaptation.

## 4.4 Robustness under Sparse Sampling

We further evaluate PAGS under sparse-view acquisition by uniformly retaining 50%, 25%, and 12.5% of the detector channels from the physical phantom measurements. This experiment tests whether the proposed representation remains stable when the inverse problem becomes increasingly underdetermined.

Fig. 5 shows that PAGS degrades gracefully as the sampling ratio decreases. Major vascular trunks and branch connectivity remain visible even at 12.5% sampling, while the main degradation appears as reduced fine-detail contrast rather than severe streak artifacts. This behavior suggests that the Gaussian PA source representation provides an efective sparse structural constraint under limited measurements.

The robustness also benefits from the propagation model. The low-order SH parameterization biases the ASoS field toward smooth path-level variations, reducing the risk of fitting sparse-view artifacts with nonphysical high-frequency SoS changes. As a result, PAGS maintains a compact source distribution and a stable propagation correction even when fewer detector channels are available. Rotating 3D views of the full-method comparison and the sparse-view results are provided in Supplementary Video 1.

## 4.5 Ablation Study and Eficiency Analysis

We isolate the two main technical components of PAGS: the analytic Gaussian acoustic projection and the SHparameterized ASoS field.

Ablation design. We use a 2 × 2 design that combines either the 10-sphere or analytic forward model with either a uniform SoS or the ASoS field:

1. Baseline: 10-sphere forward model with a uniform SoS, corresponding to vanilla SlingBAG.

2. + ASoS field: 10-sphere forward model with the SH-parameterized ASoS field.

3. + Analytic: analytic Gaussian forward model with a uniform SoS.

4. Full PAGS: analytic Gaussian forward model with the SH-parameterized ASoS field.

All variants use the same initialization, optimizer settings, random seed, and number of iterations. Adaptive density control is applied every 50 iterations, and the optimizer state is reset after each density-control step, which explains the transient spikes in the loss curves.

Fig. 6 compares the optimization behavior of the four configurations. Introducing the ASoS field substantially lowers the final signal-domain residual, indicating that the latent propagation correction provides additional flexibility for matching measurements acquired through a heterogeneous acoustic medium. Replacing the 10- sphere approximation with the analytic Gaussian model yields a similar convergence trend with markedly lower per-iteration cost. Full PAGS combines both components and reaches the lowest final signal residual.

Vanilla UBP  
Dual-SoS UBP  
SlingBAG  
Ours (PAGS)  
![](images/0ed6ee739fc95d6e43d5e7beba6f4da1894c86d67c66cd1206709a397e839193.jpg)  
Fig. 5 Robustness under sparse-view sampling on the physical phantom data, scale bar: 5 mm. PAGS is evaluated by retaining 50%, 25%, and 12.5% of the detector channels. The reconstruction degrades gradually as sampling becomes sparser, while major vascular structures and branch connectivity remain visible.

Table 2 reports the per-iteration forward–backward runtime under identical hardware and problem scale.

![](images/6f4adffe7c4eddc55ffbbc4a2fc1baccb8ffda08f1762fabf3a6cc196b563b2a.jpg)  
Fig. 6 Mean squared error (MSE) loss convergence of the four ablation configurations on the heterogeneous-SoS simulation. The ASoS field substantially reduces the final signal residual, while the analytic Gaussian forward model lowers the computational cost of each optimization iteration.

The analytic Gaussian forward model reduces runtime from 6.7 s to 2.3 s by replacing repeated error-function evaluations in the 10-sphere approximation with a single closed-form Gaussian response. Adding the ASoS field introduces only minor overhead, increasing runtime from 2.3 s to 2.4 s in the analytic setting. This shows that the proposed propagation model improves focusing while preserving most of the computational benefit of the analytic projection.

Table 2 Computational eficiency of the four ablation configurations. The analytic Gaussian forward model provides the main speedup, while adding the ASoS field introduces only minor overhead.
<table><tr><td>Configuration</td><td>Forward model</td><td>SoS field</td><td>Time/iter. (s) ↓</td></tr><tr><td>Baseline</td><td>10-sphere approx.</td><td>Uniform</td><td>6.7</td></tr><tr><td>+ ASoS field</td><td>10-sphere approx.</td><td>SH probes</td><td>6.9</td></tr><tr><td>+ Analytic</td><td>Analytic</td><td>Uniform</td><td>2.3</td></tr><tr><td>Full PAGS</td><td>Analytic</td><td>SH probes</td><td>2.4</td></tr></table>

## 5 Conclusion

We presented PAGS, a diferentiable autofocusing framework for PACT under unknown SoS heterogeneity. Instead of recovering a dense acoustic medium, PAGS learns an anisotropic path-averaged SoS field whose values directly determine source-to-transducer arrival-time alignment. This latent propagation model is coupled with sparse Gaussian PA sources and an analytic Gaussian acoustic projection, enabling signal-domain optimization of both source and propagation parameters without calibrated SoS priors. Experiments on simulated and physical phantom data show sharper reconstruction under heterogeneous acoustic media, robust behavior under sparse-view sampling, and complementary benefits from the ASoS field and the analytic projection. PAGS estimates an efective path-level propagation field rather than a physical SoS map, and the present validation is limited to simulated and phantom data. Future work will extend this path-level formulation to more complex acoustic settings and broader biological validation.

## Abbreviations

Abbrev. Definition   
3DGS 3D Gaussian splatting   
ASoS Anisotropic path-averaged speed of sound   
CT Computed tomography   
DAS Delay-and-sum   
GPU Graphics processing unit   
MSE Mean squared error   
PA Photoacoustic   
PACT Photoacoustic computed tomography   
PAGS Speed-of-sound-adaptive Gaussian splatting   
PSNR Peak signal-to-noise ratio   
RMSE Root mean square error   
SH Spherical harmonics   
SoS Speed of sound   
SSIM Structural similarity index measure   
ToF Time of flight   
TV Total variation   
UBP Universal back-projection

## Declarations

## Data Availability

The datasets generated and analyzed during the current study are available from the corresponding author on reasonable request.

## Code Availability

The oficial implementation of PAGS, together with usage instructions and a synthetic example, is publicly available at https://github.com/work-submit/ PAGS/.

## Competing Interests

The authors have no competing interests to declare that are relevant to the content of this article.

## Author Contributions

Jiarui Ge developed the methodology and algorithms, implemented the software, conducted the computational experiments, and analyzed and visualized the results. Jintao Ma, Bangxu Fan, and Jinyan Zhang performed the photoacoustic data simulations and experimental data acquisition. Xiaokang Yang provided overall super vision and strategic scientific guidance. Shuai Na and Xiaoyun Yuan jointly supervised the work, participated in regular project discussions, advised on technical details, and guided manuscript preparation and revision. All authors discussed the results, contributed to writing and revising the manuscript, and approved the final version.

## Funding

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant 62271283, the National Key Research and Development Program of China under Grant 2024YFC2421300, the Jiangsu Provincial Frontier Technology R&D Program under Grant BF2025002, and the Shanghai Jiao Tong University Medical and Engineering Cross Research Fund under Grant YG2026QNB50.

## Acknowledgements

Not applicable.

## References

1. Youzuo Lin, Shihang Feng, James Theiler, Yinpeng Chen, Umberto Villa, Jing Rao, John Greenhall, Cristian Pantea, Mark A. Anastasio, and Brendt Wohlberg. Survey of deep learning and physics-based approaches in computational wave imaging, 2024.

2. Minghua Xu and Lihong V. Wang. Universal backprojection algorithm for photoacoustic computed tomography. Physical Review E, 71(1):016706, 2005.

3. Jenni Poimala, Ben Cox, and Andreas Hauptmann. Compensating unknown speed of sound in learned fast 3d limited-view photoacoustic tomography. Photoacoustics, 37:100597, 2024.

4. Tianao Li, Manxiu Cui, Cheng Ma, and Emma Alexander. Coordinate-based speed of sound recovery for aberrationcorrected photoacoustic computed tomography. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 27466–27475, 2025.

5. Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.

6. Shuang Li, Yibing Wang, Jian Gao, Chulhong Kim, Seongwook Choi, Yu Zhang, Qian Chen, Yao Yao, and Changhui Li. Slingbag: Point cloud-based iterative algorithm for large-scale 3d photoacoustic imaging. Nature Communications, 17, 2026.

7. Xi Yang, Chengpeng Chai, Yun-Hsuan Chen, and Mohamad Sawan. Skull impact on photoacoustic imaging of multi-layered brain tissues with embedded blood vessel under diferent optical source types: Modeling and simulation. Bioengineering, 12(1):40, 2025.

8. Chao Huang, Kun Wang, Liming Nie, Lihong V. Wang, and Mark A. Anastasio. Full-wave iterative image reconstruction in photoacoustic tomography with acoustically inhomogeneous media. IEEE Transactions on Medical Imaging, 32(6):1097–1110, 2013.

9. Tong Wang, Menghui He, Kang Shen, Wen Liu, and Chao Tian. Learned regularization for image reconstruction in sparse-view photoacoustic tomography. Biomedical Optics Express, 13(11):5721–5737, 2022.

10. Xianlin Song, Wenhua Zhong, Zilong Li, Shuchong Peng, Hongyu Zhang, Guijun Wang, Jiaqing Dong, Xuan Liu, Xiaoling Xu, and Qiegen Liu. Accelerated model-based iterative reconstruction strategy for sparse-view photoacoustic tomography aided by multi-channel autoencoder priors. Journal of Biophotonics, 17(1):e202300281, 2024.

11. Richard Kowar. Time reversal for photoacoustic tomogra phy based on the wave equation of nachman, smith, and waag. Physical Review E, 89(2):023203, 2014.

12. Xianlin Song, Guijun Wang, Wenhua Zhong, Kangjun Guo, Zilong Li, Xuan Liu, Jiaqing Dong, and Qiegen Liu. Sparse-view reconstruction for photoacoustic tomography combining difusion model with model-based iteration. Photoacoustics, 33:100558, 2023.

13. Huibin Liu, Xiangyu Teng, Shuxuan Yu, Wenguang Yang, Tiantian Kong, and Tangying Liu. Recent advances in photoacoustic imaging: Current status and future perspectives. Micromachines, 15(8):1007, 2024.

14. Stephan Antholzer, Markus Haltmeier, and Johannes Schwab. Deep learning for photoacoustic tomography from sparse data. Inverse Problems in Science and Engineering, 27(7):987–1005, 2019.

15. Andreas Hauptmann, Felix Lucka, Marta Betcke, Nam Huynh, Jonas Adler, Ben Cox, Paul Beard, Sebastien Ourselin, and Simon Arridge. Model-based learning for accelerated, limited-view 3-d photoacoustic tomography. IEEE Transactions on Medical Imaging, 37(6):1382–1393, 2018.

16. Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19447–19456, 2024.

17. Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.

18. Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scafold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20654–20664, 2024.

19. Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20331– 20341, 2024.

20. Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20310–20320, 2024.

21. Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8508–8520, 2024.

22. Ruyi Zha, Tao Jun Lin, Yuanhao Cai, Jiwen Cao, Yanhao Zhang, and Hongdong Li. r<sup>2</sup>-gaussian: Rectifying radiative gaussian splatting for tomographic reconstruction. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, 2024.

23. Mark C. Eid, Ana I. L. Namburete, and Jo˜ao F. Henriques. UltraGauss: Ultrafast gaussian reconstruction of 3d ultrasound volumes. In The Fourteenth International Conference on Learning Representations (ICLR), 2026.

24. Mengxiao Geng, Zijie Chen, Ran Hong, Bingxuan Li, and Qiegen Liu. GR-Difusion: 3d gaussian representation meets difusion in whole-body PET reconstruction, 2026.

25. Kacper Marzol, Ignacy Kolton, Weronika Smolak-Dy˙zewska, Joanna Kaleta, Marcin Mazur, and Przemys law Spurek. MedGS: Gaussian splatting for multi-modal 3d medical imaging, 2025.

26. Bradley E. Treeby and Benjamin T. Cox. k-Wave: MAT-LAB toolbox for the simulation and reconstruction of photoacoustic wave fields. Journal of Biomedical Optics, 15(2):021314, 2010.